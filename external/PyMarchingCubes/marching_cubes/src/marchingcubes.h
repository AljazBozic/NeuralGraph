
#ifndef _MARCHING_CUBES_H
#define _MARCHING_CUBES_H

#include <stddef.h>
#include <array>
#include <vector>
#include <iostream>
#include <functional>
#include "Eigen/Eigen" // git clone https://gitlab.com/libeigen/eigen.git

typedef Eigen::Matrix<double, 6, 1> Vector6d;

namespace mc
{

extern int edge_table[256];
extern int triangle_table[256][16];

namespace private_
{

double mc_isovalue_interpolation(double isovalue, double f1, double f2,
    double x1, double x2);
void mc_add_vertex(double x1, double y1, double z1, double c2,
    int axis, double f1, double f2, double isovalue, std::vector<double>* vertices);



struct MC_Triangle {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	int p[3];
};

struct MC_Gridcell {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Vector3d p[8];
	double val[8];
};

struct MC_Gridcell_Color {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Vector6d p[8];
	double val[8];
};

}



template<typename vector3, typename formula>
void marching_cubes(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula f, double isovalue,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons)
{
    using coord_type = typename vector3::value_type;
    using size_type = typename vector3::size_type;
    using namespace private_;

    // Some initial checks
    if(numx < 2 || numy < 2 || numz < 2)
        return;
    
    if(!std::equal(std::begin(lower), std::end(lower), std::begin(upper),
                   [](double a, double b)->bool {return a <= b;}))
        return;
 
    
#define OWN_IMPL

#ifdef OWN_IMPL

    double dx = (upper[0] - lower[0]) / (numx - 1.0);
    double dy = (upper[1] - lower[1]) / (numy - 1.0);
    double dz = (upper[2] - lower[2]) / (numz - 1.0);

    auto coord_mapper = [&](int x, int y, int z) { return Eigen::Vector3d( lower[0] + x * dx, lower[1] + y * dy, lower[2] + z * dz ); }; 
    auto push_vertex = [&] (Eigen::Vector3d xyz) {int id = vertices.size()/3; vertices.push_back(xyz.x()); vertices.push_back(xyz.y()); vertices.push_back(xyz.z()); return id;};

    // vertex zero crossing interpolation
	//       f(p2) = valp2
	//       x
	//      /
	//     x f(p) = isolevel
	//    /
	//   /
	//  /
	// x
	// f(p1) = valp1
    auto VertexInterp = [&](double isolevel, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double valp1, double valp2) -> Eigen::Vector3d
                                {
                                    double alpha = (valp2 - isolevel) / (valp2 - valp1);
                                    return alpha*p1 + (1 - alpha)*p2;
                                };

    // store intersections of old z plane to avoid duplicated vertices
    int* edge_intersections_old_x = new int[(numx-1) * numy];
    int* edge_intersections_old_y = new int[numx * (numy-1)];
    int* edge_intersections_current_x = new int[(numx-1) * numy];
    int* edge_intersections_current_y = new int[numx * (numy-1)];

    // store intersections within the z-planes to avoid duplicated vertices
    int* edge_intersections_current_z = new int[numx * numy];

    for (int z = 0; z < numz - 1; z++)
	{
        // swap index storage
        std::swap(edge_intersections_old_x, edge_intersections_current_x); // old = current
        std::swap(edge_intersections_old_y, edge_intersections_current_y);
        std::fill_n(edge_intersections_current_x, (numx-1) * numy, -1); // invalidate
        std::fill_n(edge_intersections_current_y, (numy-1) * numx, -1); // invalidate

        std::fill_n(edge_intersections_current_z, numy * numx, -1); // invalidate

		for (int y = 0; y < numy - 1; y++)
		{
            for (int x = 0; x < numx - 1; x++)
			{
                // Process Volume Cell
                MC_Gridcell cell;
                //
                //    4---5
                //   /   /|
                //  0---1 6
                //  |   |/
                //  3---2
                // cell corners
                cell.p[0] = coord_mapper(x + 1, y, z);
                cell.p[1] = coord_mapper(x, y, z);
                cell.p[2] = coord_mapper(x, y + 1, z);
                cell.p[3] = coord_mapper(x + 1, y + 1, z);
                cell.p[4] = coord_mapper(x + 1, y, z + 1);
                cell.p[5] = coord_mapper(x, y, z + 1);
                cell.p[6] = coord_mapper(x, y + 1, z + 1);
                cell.p[7] = coord_mapper(x + 1, y + 1, z + 1);

                // cell corner values
                cell.val[0] = (double)f(x + 1, y, z);
                cell.val[1] = (double)f(x, y, z);
                cell.val[2] = (double)f(x, y + 1, z);
                cell.val[3] = (double)f(x + 1, y + 1, z);
                cell.val[4] = (double)f(x + 1, y, z + 1);
                cell.val[5] = (double)f(x, y, z + 1);
                cell.val[6] = (double)f(x, y + 1, z + 1);
                cell.val[7] = (double)f(x + 1, y + 1, z + 1);

                // triangulation code
	            int cubeindex = 0;
                if (cell.val[0] < isovalue) cubeindex |= 1;
                if (cell.val[1] < isovalue) cubeindex |= 2;
                if (cell.val[2] < isovalue) cubeindex |= 4;
                if (cell.val[3] < isovalue) cubeindex |= 8;
                if (cell.val[4] < isovalue) cubeindex |= 16;
                if (cell.val[5] < isovalue) cubeindex |= 32;
                if (cell.val[6] < isovalue) cubeindex |= 64;
                if (cell.val[7] < isovalue) cubeindex |= 128;

	            // Cube is entirely in/out of the surface
	            if (edge_table[cubeindex] == 0) continue;


                /* Find the vertices where the surface intersects the cube */
                int vertlist[12];
                { // edges on the old z plane
                    if (edge_table[cubeindex] & 1) // edge in x at y
                    {
                        if(z==0) vertlist[0] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[1], cell.val[0], cell.val[1]));
                        else vertlist[0] = edge_intersections_old_x[y * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 2) // edge in y at x
                    {
                        if(z==0) vertlist[1] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[2], cell.val[1], cell.val[2]));
                        else vertlist[1] = edge_intersections_old_y[x * (numy-1) + y];
                    }
                    if (edge_table[cubeindex] & 4) // edge in x at y+1
                    {
                        if(z==0) vertlist[2] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[3], cell.val[2], cell.val[3]));
                        else vertlist[2] = edge_intersections_old_x[(y+1) * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 8) // edge in y at x+1
                    {
                        if(z==0) vertlist[3] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[0], cell.val[3], cell.val[0]));
                        else vertlist[3] = edge_intersections_old_y[(x+1) * (numy-1) + y];
                    }
                }

                { // edges on the new z plane
                    if (edge_table[cubeindex] & 16) // edge in x at y
                    {
                        if (edge_intersections_current_x[y * (numx-1) + x] == -1) // check if already assigned
                        {
                            vertlist[4] = push_vertex(VertexInterp(isovalue, cell.p[4], cell.p[5], cell.val[4], cell.val[5]));
                            edge_intersections_current_x[y * (numx-1) + x] = vertlist[4];
                        }
                        else
                        {
                            vertlist[4] = edge_intersections_current_x[y * (numx-1) + x];
                        }
                    }
                    if (edge_table[cubeindex] & 32) // edge in y at x
                    {
                        if(edge_intersections_current_y[x * (numy-1) + y] == -1)
                        {
                            vertlist[5] = push_vertex(VertexInterp(isovalue, cell.p[5], cell.p[6], cell.val[5], cell.val[6]));
                            edge_intersections_current_y[x * (numy-1) + y] = vertlist[5];
                        }
                        else
                        {
                            vertlist[5] = edge_intersections_current_y[x * (numy-1) + y];
                        }                        
                    }
                    if (edge_table[cubeindex] & 64) // edge in x at y+1
                    {
                        if (edge_intersections_current_x[(y+1) * (numx-1) + x] == -1)
                        {
                            vertlist[6] = push_vertex(VertexInterp(isovalue, cell.p[6], cell.p[7], cell.val[6], cell.val[7]));
                            edge_intersections_current_x[(y+1) * (numx-1) + x] = vertlist[6];
                        }
                        else
                        {
                            vertlist[6] = edge_intersections_current_x[(y+1) * (numx-1) + x];
                        }                        
                    }
                    if (edge_table[cubeindex] & 128) // edge in y at x+1
                    {
                        if (edge_intersections_current_y[(x+1) * (numy-1) + y] == -1)
                        {
                            vertlist[7] = push_vertex(VertexInterp(isovalue, cell.p[7], cell.p[4], cell.val[7], cell.val[4]));
                            edge_intersections_current_y[(x+1) * (numy-1) + y] = vertlist[7];
                        }
                        else
                        {
                            vertlist[7] = edge_intersections_current_y[(x+1) * (numy-1) + y];
                        }                        
                    }
                }

                { // between the z planes
                    if (edge_table[cubeindex] & 256) // 0 -- 4,  x + 1, y
                    {
                        if (edge_intersections_current_z[y * numx + (x+1)] == -1)                    
                        {                     
                            vertlist[8] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[4], cell.val[0], cell.val[4]));
                            edge_intersections_current_z[y * numx + (x+1)] = vertlist[8];
                        }
                        else
                        {
                            vertlist[8] = edge_intersections_current_z[y * numx + (x+1)];
                        }
                    }
                    if (edge_table[cubeindex] & 512) // 1 -- 5,  x, y
                    {
                        if (edge_intersections_current_z[y * numx + x] == -1)                    
                        { 
                            vertlist[9] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[5], cell.val[1], cell.val[5]));
                            edge_intersections_current_z[y * numx + x] = vertlist[9];
                        }
                        else
                        {
                            vertlist[9] = edge_intersections_current_z[y * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 1024) // 2 -- 6,  x, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + x] == -1)                    
                        { 
                            vertlist[10] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[6], cell.val[2], cell.val[6]));
                            edge_intersections_current_z[(y+1) * numx + x] = vertlist[10];
                        }
                        else
                        {
                            vertlist[10] = edge_intersections_current_z[(y+1) * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 2048) // 3 -- 7,  x + 1, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + (x+1)] == -1)                    
                        { 
                            vertlist[11] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[7], cell.val[3], cell.val[7]));
                            edge_intersections_current_z[(y+1) * numx + (x+1)] = vertlist[11];
                        }
                        else
                        {
                            vertlist[11] = edge_intersections_current_z[(y+1) * numx + (x+1)];
                        }
                    }
                }

                // push face indices
                for (int i = 0; triangle_table[cubeindex][i] != -1; ++i)
                    polygons.push_back(vertlist[triangle_table[cubeindex][i]]);

			}
		}
	}


    delete[] edge_intersections_old_x;
    delete[] edge_intersections_old_y;
    delete[] edge_intersections_current_x;
    delete[] edge_intersections_current_y;
    delete[] edge_intersections_current_z;

#else

    // numx, numy and numz are the numbers of evaluations in each direction
    --numx; --numy; --numz;
    
    coord_type dx = (upper[0] - lower[0]) / static_cast<coord_type>(numx);
    coord_type dy = (upper[1] - lower[1]) / static_cast<coord_type>(numy);
    coord_type dz = (upper[2] - lower[2]) / static_cast<coord_type>(numz);
    
    const int num_shared_indices = 2 * (numy + 1) * (numz + 1);
    std::vector<size_type> shared_indices_x(num_shared_indices);
    std::vector<size_type> shared_indices_y(num_shared_indices);
    std::vector<size_type> shared_indices_z(num_shared_indices);
    
    // const int numz = numz*3;
    const int numyz = numy*numz;
    
    for(int i=0; i<numx; ++i)
    {
        coord_type x = lower[0] + dx*i;
        coord_type x_dx = lower[0] + dx*(i+1);
        const int i_mod_2 = i % 2;
        const int i_mod_2_inv = (i_mod_2 ? 0 : 1); 

        for(int j=0; j<numy; ++j)
        {
            coord_type y = lower[1] + dy*j;
            coord_type y_dy = lower[1] + dy*(j+1);

            double v[8];
            v[4] = f(x, y, lower[2]); v[5] = f(x_dx, y, lower[2]);
            v[6] = f(x_dx, y_dy, lower[2]); v[7] = f(x, y_dy, lower[2]);

            for(int k=0; k<numz; ++k)
            {
                coord_type z = lower[2] + dz*k;
                coord_type z_dz = lower[2] + dz*(k+1);
                
                v[0] = v[4]; v[1] = v[5];
                v[2] = v[6]; v[3] = v[7];
                v[4] = f(x, y, z_dz); v[5] = f(x_dx, y, z_dz);
                v[6] = f(x_dx, y_dy, z_dz); v[7] = f(x, y_dy, z_dz);
                
                unsigned int cubeindex = 0;
                for(int m=0; m<8; ++m)
                    if(v[m] <= isovalue)
                        cubeindex |= 1<<m;
                
                // Generate vertices AVOIDING DUPLICATES.
                
                int edges = edge_table[cubeindex];
                std::array<size_type, 12> indices;
                // for three edges we have to compute it for sure (if zerocrossing)
                if(edges & 0x040)
                {
                    indices[6] = vertices.size() / 3;
                    shared_indices_x[i_mod_2_inv*numyz + (j+1)*numz + (k+1)] = indices[6];
                    mc_add_vertex(x_dx, y_dy, z_dz, x, 0, v[6], v[7], isovalue, &vertices);
                }
                if(edges & 0x020)
                {
                    indices[5] = vertices.size() / 3;
                    shared_indices_y[i_mod_2_inv*numyz + (j+1)*numz + (k+1)] = indices[5];
                    mc_add_vertex(x_dx, y, z_dz, y_dy, 1, v[5], v[6], isovalue, &vertices);
                }
                if(edges & 0x400)
                {
                    indices[10] = vertices.size() / 3;
                    shared_indices_z[i_mod_2_inv*numyz + (j+1)*numz + (k+1)] = indices[10];
                    mc_add_vertex(x_dx, y+dx, z, z_dz, 2, v[2], v[6], isovalue, &vertices);
                }
                
                if(edges & 0x001)
                {
                    if(j == 0 && k == 0)
                    {
                        indices[0] = vertices.size() / 3;
                        mc_add_vertex(x, y, z, x_dx, 0, v[0], v[1], isovalue, &vertices);
                    }
                    else
                        indices[0] = shared_indices_x[i_mod_2_inv*numyz + j*numz + k];
                }
                if(edges & 0x002)
                {
                    if(k == 0)
                    {
                        indices[1] = vertices.size() / 3;
                        shared_indices_y[i_mod_2_inv*numyz + (j+1)*numz + k] = indices[1];
                        mc_add_vertex(x_dx, y, z, y_dy, 1, v[1], v[2], isovalue, &vertices);
                    }
                    else
                        indices[1] = shared_indices_y[i_mod_2_inv*numyz + (j+1)*numz + k];
                }
                if(edges & 0x004)
                {
                    if(k == 0)
                    {
                        indices[2] = vertices.size() / 3;
                        shared_indices_x[i_mod_2_inv*numyz + (j+1)*numz + k] = indices[2];
                        mc_add_vertex(x_dx, y_dy, z, x, 0, v[2], v[3], isovalue, &vertices);
                    }
                    else
                        indices[2] = shared_indices_x[i_mod_2_inv*numyz + (j+1)*numz + k];
                }
                if(edges & 0x008)
                {
                    if(i == 0 && k == 0)
                    {
                        indices[3] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, y, 1, v[3], v[0], isovalue, &vertices);
                    }
                    else
                        indices[3] = shared_indices_y[i_mod_2*numyz + (j+1)*numz + k];
                }
                if(edges & 0x010)
                {
                    if(j == 0)
                    {
                        indices[4] = vertices.size() / 3;
                        shared_indices_x[i_mod_2_inv*numyz + j*numz + (k+1)] = indices[4];
                        mc_add_vertex(x, y, z_dz, x_dx, 0, v[4], v[5], isovalue, &vertices);
                    }
                    else
                        indices[4] = shared_indices_x[i_mod_2_inv*numyz + j*numz + (k+1)];
                }
                if(edges & 0x080)
                {
                    if(i == 0)
                    {
                        indices[7] = vertices.size() / 3;
                        shared_indices_y[i_mod_2*numyz + (j+1)*numz + (k+1)] = indices[7];
                        mc_add_vertex(x, y_dy, z_dz, y, 1, v[7], v[4], isovalue, &vertices);
                    }
                    else
                        indices[7] = shared_indices_y[i_mod_2*numyz + (j+1)*numz + (k+1)];
                }
                if(edges & 0x100)
                {
                    if(i == 0 && j == 0)
                    {
                        indices[8] = vertices.size() / 3;
                        mc_add_vertex(x, y, z, z_dz, 2, v[0], v[4], isovalue, &vertices);
                    }
                    else
                        indices[8] = shared_indices_z[i_mod_2*numyz + j*numz + (k+1)];
                }
                if(edges & 0x200)
                {
                    if(j == 0)
                    {
                        indices[9] = vertices.size() / 3;
                        shared_indices_z[i_mod_2_inv*numyz + j*numz + (k+1)] = indices[9];
                        mc_add_vertex(x_dx, y, z, z_dz, 2, v[1], v[5], isovalue, &vertices);
                    }
                    else
                        indices[9] = shared_indices_z[i_mod_2_inv*numyz + j*numz + (k+1)];
                }
                if(edges & 0x800)
                {
                    if(i == 0)
                    {
                        indices[11] = vertices.size() / 3;
                        shared_indices_z[i_mod_2*numyz + (j+1)*numz + (k+1)] = indices[11];
                        mc_add_vertex(x, y_dy, z, z_dz, 2, v[3], v[7], isovalue, &vertices);
                    }
                    else
                        indices[11] = shared_indices_z[i_mod_2*numyz + (j+1)*numz + (k+1)];
                }
                
                int tri;
                int* triangle_table_ptr = triangle_table[cubeindex];
                for(int m=0; tri = triangle_table_ptr[m], tri != -1; ++m)
                    polygons.push_back(indices[tri]);


                /*if(triangle_table_ptr[0] == -1) continue;
                Eigen::Vector3d last(vertices[3*indices[triangle_table_ptr[0] ]+0], vertices[3*indices[triangle_table_ptr[0] ]+1], vertices[3*indices[triangle_table_ptr[0] ]+2]);
                for(int m=0; tri = triangle_table_ptr[m], tri != -1; ++m)
                {
                    Eigen::Vector3d current(vertices[3*indices[tri]+0], vertices[3*indices[tri]+1], vertices[3*indices[tri]+2]);
                    if ((current-last).norm() > std::sqrt(dx*dx + dy*dy + dz*dz))
                    {
                        std::cout << "error: " << (current-last).norm() << std::endl;
                        std::cout << "cubeindex: " << cubeindex << std::endl;
                        std::cout << "i: " << i << std::endl;
                        std::cout << "j: " << j << std::endl;
                        std::cout << "k: " << k << std::endl;
                        std::cout << "m: " << m << std::endl;
                    }
                    last = current;
                    polygons.push_back(indices[tri]);
                }*/
            }
        }
    }
    #endif
    
}




/////////////////////////////
/////////////////////////////
//////////   COLOR //////////
/////////////////////////////
/////////////////////////////



template<typename vector3, typename formula_sdf, typename formula_color>
void marching_cubes_color(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula_sdf f_sdf, formula_color f_color, double isovalue,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons)
{
    using coord_type = typename vector3::value_type;
    using size_type = typename vector3::size_type;
    using namespace private_;

    // Some initial checks
    if(numx < 2 || numy < 2 || numz < 2)
        return;
    
    if(!std::equal(std::begin(lower), std::end(lower), std::begin(upper),
                   [](double a, double b)->bool {return a <= b;}))
        return;
 
    
    double dx = (upper[0] - lower[0]) / (numx - 1.0);
    double dy = (upper[1] - lower[1]) / (numy - 1.0);
    double dz = (upper[2] - lower[2]) / (numz - 1.0);

    auto coord_mapper = [&](int x, int y, int z) { return Eigen::Vector3d( lower[0] + x * dx, lower[1] + y * dy, lower[2] + z * dz ); }; 
    //auto push_vertex = [&] (Eigen::Vector3d xyz) {int id = vertices.size()/3; vertices.push_back(xyz.x()); vertices.push_back(xyz.y()); vertices.push_back(xyz.z()); return id;};
    auto push_vertex = [&] (const Vector6d& xyz_rgb)
                                {
                                    int id = vertices.size()/6;
                                    for(unsigned int i=0; i<6; ++i) vertices.push_back(xyz_rgb[i]);
                                    return id;
                                };

    // vertex zero crossing interpolation
	//       f(p2) = valp2
	//       x
	//      /
	//     x f(p) = isolevel
	//    /
	//   /
	//  /
	// x
	// f(p1) = valp1
    auto VertexInterp = [&](double isolevel, const Vector6d& p1, const Vector6d& p2, double valp1, double valp2) -> Vector6d
                                {
                                    double alpha = (valp2 - isolevel) / (valp2 - valp1);
                                    return alpha*p1 + (1 - alpha)*p2;
                                };

    // store intersections of old z plane to avoid duplicated vertices
    int* edge_intersections_old_x = new int[(numx-1) * numy];
    int* edge_intersections_old_y = new int[numx * (numy-1)];
    int* edge_intersections_current_x = new int[(numx-1) * numy];
    int* edge_intersections_current_y = new int[numx * (numy-1)];

    // store intersections within the z-planes to avoid duplicated vertices
    int* edge_intersections_current_z = new int[numx * numy];

    for (int z = 0; z < numz - 1; z++)
	{
        // swap index storage
        std::swap(edge_intersections_old_x, edge_intersections_current_x); // old = current
        std::swap(edge_intersections_old_y, edge_intersections_current_y);
        std::fill_n(edge_intersections_current_x, (numx-1) * numy, -1); // invalidate
        std::fill_n(edge_intersections_current_y, (numy-1) * numx, -1); // invalidate

        std::fill_n(edge_intersections_current_z, numy * numx, -1); // invalidate

		for (int y = 0; y < numy - 1; y++)
		{
            for (int x = 0; x < numx - 1; x++)
			{
                // Process Volume Cell
                MC_Gridcell_Color cell;
                //
                //    4---5
                //   /   /|
                //  0---1 6
                //  |   |/
                //  3---2
                // cell corners
                cell.p[0].block<3,1>(0,0) = coord_mapper(x + 1, y, z);
                cell.p[1].block<3,1>(0,0) = coord_mapper(x, y, z);
                cell.p[2].block<3,1>(0,0) = coord_mapper(x, y + 1, z);
                cell.p[3].block<3,1>(0,0) = coord_mapper(x + 1, y + 1, z);
                cell.p[4].block<3,1>(0,0) = coord_mapper(x + 1, y, z + 1);
                cell.p[5].block<3,1>(0,0) = coord_mapper(x, y, z + 1);
                cell.p[6].block<3,1>(0,0) = coord_mapper(x, y + 1, z + 1);
                cell.p[7].block<3,1>(0,0) = coord_mapper(x + 1, y + 1, z + 1);

                // cell colors
                cell.p[0].block<3,1>(3,0) = f_color(x + 1, y, z);
                cell.p[1].block<3,1>(3,0) = f_color(x, y, z);
                cell.p[2].block<3,1>(3,0) = f_color(x, y + 1, z);
                cell.p[3].block<3,1>(3,0) = f_color(x + 1, y + 1, z);
                cell.p[4].block<3,1>(3,0) = f_color(x + 1, y, z + 1);
                cell.p[5].block<3,1>(3,0) = f_color(x, y, z + 1);
                cell.p[6].block<3,1>(3,0) = f_color(x, y + 1, z + 1);
                cell.p[7].block<3,1>(3,0) = f_color(x + 1, y + 1, z + 1);

                // cell corner values
                cell.val[0] = (double)f_sdf(x + 1, y, z);
                cell.val[1] = (double)f_sdf(x, y, z);
                cell.val[2] = (double)f_sdf(x, y + 1, z);
                cell.val[3] = (double)f_sdf(x + 1, y + 1, z);
                cell.val[4] = (double)f_sdf(x + 1, y, z + 1);
                cell.val[5] = (double)f_sdf(x, y, z + 1);
                cell.val[6] = (double)f_sdf(x, y + 1, z + 1);
                cell.val[7] = (double)f_sdf(x + 1, y + 1, z + 1);

                // triangulation code
	            int cubeindex = 0;
                if (cell.val[0] < isovalue) cubeindex |= 1;
                if (cell.val[1] < isovalue) cubeindex |= 2;
                if (cell.val[2] < isovalue) cubeindex |= 4;
                if (cell.val[3] < isovalue) cubeindex |= 8;
                if (cell.val[4] < isovalue) cubeindex |= 16;
                if (cell.val[5] < isovalue) cubeindex |= 32;
                if (cell.val[6] < isovalue) cubeindex |= 64;
                if (cell.val[7] < isovalue) cubeindex |= 128;

	            // Cube is entirely in/out of the surface
	            if (edge_table[cubeindex] == 0) continue;


                /* Find the vertices where the surface intersects the cube */
                int vertlist[12];
                { // edges on the old z plane
                    if (edge_table[cubeindex] & 1) // edge in x at y
                    {
                        if(z==0) vertlist[0] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[1], cell.val[0], cell.val[1]));
                        else vertlist[0] = edge_intersections_old_x[y * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 2) // edge in y at x
                    {
                        if(z==0) vertlist[1] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[2], cell.val[1], cell.val[2]));
                        else vertlist[1] = edge_intersections_old_y[x * (numy-1) + y];
                    }
                    if (edge_table[cubeindex] & 4) // edge in x at y+1
                    {
                        if(z==0) vertlist[2] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[3], cell.val[2], cell.val[3]));
                        else vertlist[2] = edge_intersections_old_x[(y+1) * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 8) // edge in y at x+1
                    {
                        if(z==0) vertlist[3] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[0], cell.val[3], cell.val[0]));
                        else vertlist[3] = edge_intersections_old_y[(x+1) * (numy-1) + y];
                    }
                }

                { // edges on the new z plane
                    if (edge_table[cubeindex] & 16) // edge in x at y
                    {
                        if (edge_intersections_current_x[y * (numx-1) + x] == -1) // check if already assigned
                        {
                            vertlist[4] = push_vertex(VertexInterp(isovalue, cell.p[4], cell.p[5], cell.val[4], cell.val[5]));
                            edge_intersections_current_x[y * (numx-1) + x] = vertlist[4];
                        }
                        else
                        {
                            vertlist[4] = edge_intersections_current_x[y * (numx-1) + x];
                        }
                    }
                    if (edge_table[cubeindex] & 32) // edge in y at x
                    {
                        if(edge_intersections_current_y[x * (numy-1) + y] == -1)
                        {
                            vertlist[5] = push_vertex(VertexInterp(isovalue, cell.p[5], cell.p[6], cell.val[5], cell.val[6]));
                            edge_intersections_current_y[x * (numy-1) + y] = vertlist[5];
                        }
                        else
                        {
                            vertlist[5] = edge_intersections_current_y[x * (numy-1) + y];
                        }                        
                    }
                    if (edge_table[cubeindex] & 64) // edge in x at y+1
                    {
                        if (edge_intersections_current_x[(y+1) * (numx-1) + x] == -1)
                        {
                            vertlist[6] = push_vertex(VertexInterp(isovalue, cell.p[6], cell.p[7], cell.val[6], cell.val[7]));
                            edge_intersections_current_x[(y+1) * (numx-1) + x] = vertlist[6];
                        }
                        else
                        {
                            vertlist[6] = edge_intersections_current_x[(y+1) * (numx-1) + x];
                        }                        
                    }
                    if (edge_table[cubeindex] & 128) // edge in y at x+1
                    {
                        if (edge_intersections_current_y[(x+1) * (numy-1) + y] == -1)
                        {
                            vertlist[7] = push_vertex(VertexInterp(isovalue, cell.p[7], cell.p[4], cell.val[7], cell.val[4]));
                            edge_intersections_current_y[(x+1) * (numy-1) + y] = vertlist[7];
                        }
                        else
                        {
                            vertlist[7] = edge_intersections_current_y[(x+1) * (numy-1) + y];
                        }                        
                    }
                }

                { // between the z planes
                    if (edge_table[cubeindex] & 256) // 0 -- 4,  x + 1, y
                    {
                        if (edge_intersections_current_z[y * numx + (x+1)] == -1)                    
                        {                     
                            vertlist[8] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[4], cell.val[0], cell.val[4]));
                            edge_intersections_current_z[y * numx + (x+1)] = vertlist[8];
                        }
                        else
                        {
                            vertlist[8] = edge_intersections_current_z[y * numx + (x+1)];
                        }
                    }
                    if (edge_table[cubeindex] & 512) // 1 -- 5,  x, y
                    {
                        if (edge_intersections_current_z[y * numx + x] == -1)                    
                        { 
                            vertlist[9] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[5], cell.val[1], cell.val[5]));
                            edge_intersections_current_z[y * numx + x] = vertlist[9];
                        }
                        else
                        {
                            vertlist[9] = edge_intersections_current_z[y * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 1024) // 2 -- 6,  x, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + x] == -1)                    
                        { 
                            vertlist[10] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[6], cell.val[2], cell.val[6]));
                            edge_intersections_current_z[(y+1) * numx + x] = vertlist[10];
                        }
                        else
                        {
                            vertlist[10] = edge_intersections_current_z[(y+1) * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 2048) // 3 -- 7,  x + 1, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + (x+1)] == -1)                    
                        { 
                            vertlist[11] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[7], cell.val[3], cell.val[7]));
                            edge_intersections_current_z[(y+1) * numx + (x+1)] = vertlist[11];
                        }
                        else
                        {
                            vertlist[11] = edge_intersections_current_z[(y+1) * numx + (x+1)];
                        }
                    }
                }

                // push face indices
                for (int i = 0; triangle_table[cubeindex][i] != -1; ++i)
                    polygons.push_back(vertlist[triangle_table[cubeindex][i]]);

			}
		}
	}


    delete[] edge_intersections_old_x;
    delete[] edge_intersections_old_y;
    delete[] edge_intersections_current_x;
    delete[] edge_intersections_current_y;
    delete[] edge_intersections_current_z;
    
}




//////////////////////////////////////
//////////////////////////////////////
//////////  SUPER SAMPLING  //////////
//////////////////////////////////////
//////////////////////////////////////


template<typename vector3, typename formula, typename formulaX, typename formulaY, typename formulaZ>
void marching_cubes_super_sampling(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, 
    int superx, int supery, int superz, 
    formula f, 
    formulaX f_superX, formulaY f_superY, formulaZ f_superZ, 
    double isovalue,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons)
{
    using coord_type = typename vector3::value_type;
    using size_type = typename vector3::size_type;
    using namespace private_;

    // Some initial checks
    if(numx < 2 || numy < 2 || numz < 2)
        return;
    
    if(!std::equal(std::begin(lower), std::end(lower), std::begin(upper),
                   [](double a, double b)->bool {return a <= b;}))
        return;
 

    double dx = (upper[0] - lower[0]) / (numx - 1.0);
    double dy = (upper[1] - lower[1]) / (numy - 1.0);
    double dz = (upper[2] - lower[2]) / (numz - 1.0);



    double scale_x = (numx - 1.0) / ( numx + (numx - 1) * superx - 1.0 ); // map super sampling coordinates back to actual sampling coordinates
    double scale_y = (numy - 1.0) / ( numy + (numy - 1) * supery - 1.0 );
    double scale_z = (numz - 1.0) / ( numz + (numz - 1) * superz - 1.0 );

    //auto coord_mapper = [&](int x, int y, int z) { return Eigen::Vector3d( lower[0] + x * dx, lower[1] + y * dy, lower[2] + z * dz ); }; 
    auto coord_mapper = [&](int x, int y, int z) { return Eigen::Vector3d( x,y,z ); }; 
    auto push_vertex = [&] (Eigen::Vector3d xyz) {int id = vertices.size()/3; vertices.push_back(xyz.x()); vertices.push_back(xyz.y()); vertices.push_back(xyz.z()); return id;};

    auto sign = [](double x) { return x > 0.0; };

    // vertex zero crossing interpolation
	//       f(p2) = valp2
	//       x
	//      /
	//     x f(p) = isolevel
	//    /
	//   /
	//  /
	// x
	// f(p1) = valp1
    auto VertexInterp = [&](double isolevel, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double valp1, double valp2) -> Eigen::Vector3d
                                {
                                    bool edge_in_x_dir = p1.x() != p2.x() && p1.y() == p2.y() && p1.z() == p2.z();
                                    bool edge_in_y_dir = p1.x() == p2.x() && p1.y() != p2.y() && p1.z() == p2.z();
                                    bool edge_in_z_dir = p1.x() == p2.x() && p1.y() == p2.y() && p1.z() != p2.z();
                                    //if (edge_in_x_dir == true && edge_in_y_dir==true) std::cout << "ERROR XY" << std::endl;
                                    //if (edge_in_x_dir == true && edge_in_z_dir==true) std::cout << "ERROR XZ" << std::endl;
                                    //if (edge_in_y_dir == true && edge_in_z_dir==true) std::cout << "ERROR YZ" << std::endl;
                                    //if (edge_in_x_dir == false && edge_in_y_dir == false && edge_in_z_dir==false) std::cout << "ERROR XYZ=FALSE" << std::endl;
                                    Eigen::Vector3d p1_ss = p1;
                                    Eigen::Vector3d p2_ss = p2;
                                    double valp1_ss = valp1;
                                    double valp2_ss = valp2;

                                    //if (edge_in_x_dir) std::cout << "edge_in_x_dir" << std::endl;
                                    //if (edge_in_y_dir) std::cout << "edge_in_y_dir" << std::endl;
                                    //if (edge_in_z_dir) std::cout << "edge_in_z_dir" << std::endl;

                                    //if (fabs(valp1-isolevel)!=0.0 && fabs(valp2-isolevel)==0.0)
                                    {
                                        if (edge_in_x_dir)
                                        {
                                            // [ 0.0, 0.0, 0.0, 1.0, 1.0  ] -->valp1 = 0.0, valp2=1.0 --> val_prev=0.0
                                            // [ 1.0, 1.0, 0.0, 0.0, 0.0  ] -->valp1 = 1.0, valp2=0.0 --> val_prev=1.0
                                            int min_x = std::min(p1.x(), p2.x());
                                            double val_prev = valp1;
                                            if(p2.x() == min_x) val_prev=valp2;
                                            int y = p1.y();
                                            int z = p1.z();
                                            for(int i=1; i<=superx+1; ++i) // could do interval halfing / binary search
                                            {
                                                // find isolevel point
                                                int x = min_x*(superx+1) + i;
                                                double val = (double)f_superX(x, y, z);
                                                if (sign(val-isolevel) != sign(val_prev-isolevel) || (fabs(val-isolevel)==0.0 && fabs(val_prev-isolevel)!=0.0 )) // zero crossing
                                                {
                                                    valp1_ss = val_prev;
                                                    valp2_ss = val;
                                                    p1_ss.x() = (x - 1) * scale_x;
                                                    p2_ss.x() = x * scale_x;
                                                    break;
                                                }
                                                val_prev = val;
                                            }
                                        }

                                        if (edge_in_y_dir)
                                        {
                                            int min_y = std::min(p1.y(), p2.y());
                                            double val_prev = valp1;
                                            if(p2.y() == min_y) val_prev=valp2;
                                            int x = p1.x();
                                            int z = p1.z();
                                            for(int i=1; i<=supery+1; ++i) // could do interval halfing / binary search
                                            {
                                                // find isolevel point
                                                int y = min_y*(supery+1) + i;
                                                double val = (double)f_superY(x, y, z);
                                                if (sign(val-isolevel) != sign(val_prev-isolevel) || (fabs(val-isolevel)==0.0 && fabs(val_prev-isolevel)!=0.0 )) // zero crossing
                                                {
                                                    valp1_ss = val_prev;
                                                    valp2_ss = val;
                                                    p1_ss.y() = (y - 1) * scale_y;
                                                    p2_ss.y() = y * scale_y;
                                                    break;
                                                }
                                                val_prev = val;
                                            }
                                        }

                                        if (edge_in_z_dir)
                                        {
                                            int min_z = std::min(p1.z(), p2.z());
                                            double val_prev = valp1;
                                            if(p2.z() == min_z) val_prev=valp2;
                                            int x = p1.x();
                                            int y = p1.y();
                                            for(int i=1; i<=superz+1; ++i) // could do interval halfing / binary search
                                            {
                                                // find isolevel point
                                                int z = min_z*(superz+1) + i;
                                                double val = (double)f_superZ(x, y, z);
                                                if (sign(val-isolevel) != sign(val_prev-isolevel) || (fabs(val-isolevel)==0.0 && fabs(val_prev-isolevel)!=0.0 )) // zero crossing
                                                {
                                                    valp1_ss = val_prev;
                                                    valp2_ss = val;
                                                    p1_ss.z() = (z - 1) * scale_z;
                                                    p2_ss.z() = z * scale_z;
                                                    break;
                                                }
                                                val_prev = val;
                                            }
                                        }
                                    }

                                    // coord mapper
                                    p1_ss = Eigen::Vector3d( lower[0] + p1_ss.x() * dx, lower[1] + p1_ss.y() * dy, lower[2] + p1_ss.z() * dz);
                                    p2_ss = Eigen::Vector3d( lower[0] + p2_ss.x() * dx, lower[1] + p2_ss.y() * dy, lower[2] + p2_ss.z() * dz);

                                    // actual interpolation
                                    double alpha = (valp2_ss - isolevel) / (valp2_ss - valp1_ss);
                                    if (valp2_ss - valp1_ss == 0.0)
                                    {
                                        alpha = 0.5;
                                    }
                                    return alpha*p1_ss + (1 - alpha)*p2_ss;
                                    
                                };

    // store intersections of old z plane to avoid duplicated vertices
    int* edge_intersections_old_x = new int[(numx-1) * numy];
    int* edge_intersections_old_y = new int[numx * (numy-1)];
    int* edge_intersections_current_x = new int[(numx-1) * numy];
    int* edge_intersections_current_y = new int[numx * (numy-1)];

    // store intersections within the z-planes to avoid duplicated vertices
    int* edge_intersections_current_z = new int[numx * numy];

    for (int z = 0; z < numz - 1; z++)
	{
        // swap index storage
        std::swap(edge_intersections_old_x, edge_intersections_current_x); // old = current
        std::swap(edge_intersections_old_y, edge_intersections_current_y);
        std::fill_n(edge_intersections_current_x, (numx-1) * numy, -1); // invalidate
        std::fill_n(edge_intersections_current_y, (numy-1) * numx, -1); // invalidate

        std::fill_n(edge_intersections_current_z, numy * numx, -1); // invalidate

		for (int y = 0; y < numy - 1; y++)
		{
            for (int x = 0; x < numx - 1; x++)
			{
                // Process Volume Cell
                MC_Gridcell cell;
                //
                //    4---5
                //   /   /|
                //  0---1 6
                //  |   |/
                //  3---2
                // cell corners
                cell.p[0] = coord_mapper(x + 1, y, z);
                cell.p[1] = coord_mapper(x, y, z);
                cell.p[2] = coord_mapper(x, y + 1, z);
                cell.p[3] = coord_mapper(x + 1, y + 1, z);
                cell.p[4] = coord_mapper(x + 1, y, z + 1);
                cell.p[5] = coord_mapper(x, y, z + 1);
                cell.p[6] = coord_mapper(x, y + 1, z + 1);
                cell.p[7] = coord_mapper(x + 1, y + 1, z + 1);

                // cell corner values
                cell.val[0] = (double)f(x + 1, y, z);
                cell.val[1] = (double)f(x, y, z);
                cell.val[2] = (double)f(x, y + 1, z);
                cell.val[3] = (double)f(x + 1, y + 1, z);
                cell.val[4] = (double)f(x + 1, y, z + 1);
                cell.val[5] = (double)f(x, y, z + 1);
                cell.val[6] = (double)f(x, y + 1, z + 1);
                cell.val[7] = (double)f(x + 1, y + 1, z + 1);

                // triangulation code
	            int cubeindex = 0;
                if (cell.val[0] < isovalue) cubeindex |= 1;
                if (cell.val[1] < isovalue) cubeindex |= 2;
                if (cell.val[2] < isovalue) cubeindex |= 4;
                if (cell.val[3] < isovalue) cubeindex |= 8;
                if (cell.val[4] < isovalue) cubeindex |= 16;
                if (cell.val[5] < isovalue) cubeindex |= 32;
                if (cell.val[6] < isovalue) cubeindex |= 64;
                if (cell.val[7] < isovalue) cubeindex |= 128;

	            // Cube is entirely in/out of the surface
	            if (edge_table[cubeindex] == 0) continue;


                /* Find the vertices where the surface intersects the cube */
                int vertlist[12];
                { // edges on the old z plane
                    if (edge_table[cubeindex] & 1) // edge in x at y
                    {
                        if(z==0) vertlist[0] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[1], cell.val[0], cell.val[1]));
                        else vertlist[0] = edge_intersections_old_x[y * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 2) // edge in y at x
                    {
                        if(z==0) vertlist[1] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[2], cell.val[1], cell.val[2]));
                        else vertlist[1] = edge_intersections_old_y[x * (numy-1) + y];
                    }
                    if (edge_table[cubeindex] & 4) // edge in x at y+1
                    {
                        if(z==0) vertlist[2] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[3], cell.val[2], cell.val[3]));
                        else vertlist[2] = edge_intersections_old_x[(y+1) * (numx-1) + x];
                    }
                    if (edge_table[cubeindex] & 8) // edge in y at x+1
                    {
                        if(z==0) vertlist[3] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[0], cell.val[3], cell.val[0]));
                        else vertlist[3] = edge_intersections_old_y[(x+1) * (numy-1) + y];
                    }
                }

                { // edges on the new z plane
                    if (edge_table[cubeindex] & 16) // edge in x at y
                    {
                        if (edge_intersections_current_x[y * (numx-1) + x] == -1) // check if already assigned
                        {
                            vertlist[4] = push_vertex(VertexInterp(isovalue, cell.p[4], cell.p[5], cell.val[4], cell.val[5]));
                            edge_intersections_current_x[y * (numx-1) + x] = vertlist[4];
                        }
                        else
                        {
                            vertlist[4] = edge_intersections_current_x[y * (numx-1) + x];
                        }
                    }
                    if (edge_table[cubeindex] & 32) // edge in y at x
                    {
                        if(edge_intersections_current_y[x * (numy-1) + y] == -1)
                        {
                            vertlist[5] = push_vertex(VertexInterp(isovalue, cell.p[5], cell.p[6], cell.val[5], cell.val[6]));
                            edge_intersections_current_y[x * (numy-1) + y] = vertlist[5];
                        }
                        else
                        {
                            vertlist[5] = edge_intersections_current_y[x * (numy-1) + y];
                        }                        
                    }
                    if (edge_table[cubeindex] & 64) // edge in x at y+1
                    {
                        if (edge_intersections_current_x[(y+1) * (numx-1) + x] == -1)
                        {
                            vertlist[6] = push_vertex(VertexInterp(isovalue, cell.p[6], cell.p[7], cell.val[6], cell.val[7]));
                            edge_intersections_current_x[(y+1) * (numx-1) + x] = vertlist[6];
                        }
                        else
                        {
                            vertlist[6] = edge_intersections_current_x[(y+1) * (numx-1) + x];
                        }                        
                    }
                    if (edge_table[cubeindex] & 128) // edge in y at x+1
                    {
                        if (edge_intersections_current_y[(x+1) * (numy-1) + y] == -1)
                        {
                            vertlist[7] = push_vertex(VertexInterp(isovalue, cell.p[7], cell.p[4], cell.val[7], cell.val[4]));
                            edge_intersections_current_y[(x+1) * (numy-1) + y] = vertlist[7];
                        }
                        else
                        {
                            vertlist[7] = edge_intersections_current_y[(x+1) * (numy-1) + y];
                        }                        
                    }
                }

                { // between the z planes
                    if (edge_table[cubeindex] & 256) // 0 -- 4,  x + 1, y
                    {
                        if (edge_intersections_current_z[y * numx + (x+1)] == -1)                    
                        {                     
                            vertlist[8] = push_vertex(VertexInterp(isovalue, cell.p[0], cell.p[4], cell.val[0], cell.val[4]));
                            edge_intersections_current_z[y * numx + (x+1)] = vertlist[8];
                        }
                        else
                        {
                            vertlist[8] = edge_intersections_current_z[y * numx + (x+1)];
                        }
                    }
                    if (edge_table[cubeindex] & 512) // 1 -- 5,  x, y
                    {
                        if (edge_intersections_current_z[y * numx + x] == -1)                    
                        { 
                            vertlist[9] = push_vertex(VertexInterp(isovalue, cell.p[1], cell.p[5], cell.val[1], cell.val[5]));
                            edge_intersections_current_z[y * numx + x] = vertlist[9];
                        }
                        else
                        {
                            vertlist[9] = edge_intersections_current_z[y * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 1024) // 2 -- 6,  x, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + x] == -1)                    
                        { 
                            vertlist[10] = push_vertex(VertexInterp(isovalue, cell.p[2], cell.p[6], cell.val[2], cell.val[6]));
                            edge_intersections_current_z[(y+1) * numx + x] = vertlist[10];
                        }
                        else
                        {
                            vertlist[10] = edge_intersections_current_z[(y+1) * numx + x];
                        }
                    }
                    if (edge_table[cubeindex] & 2048) // 3 -- 7,  x + 1, y + 1
                    {
                        if (edge_intersections_current_z[(y+1) * numx + (x+1)] == -1)                    
                        { 
                            vertlist[11] = push_vertex(VertexInterp(isovalue, cell.p[3], cell.p[7], cell.val[3], cell.val[7]));
                            edge_intersections_current_z[(y+1) * numx + (x+1)] = vertlist[11];
                        }
                        else
                        {
                            vertlist[11] = edge_intersections_current_z[(y+1) * numx + (x+1)];
                        }
                    }
                }

                // push face indices
                for (int i = 0; triangle_table[cubeindex][i] != -1; ++i)
                    polygons.push_back(vertlist[triangle_table[cubeindex][i]]);

			}
		}
	}


    delete[] edge_intersections_old_x;
    delete[] edge_intersections_old_y;
    delete[] edge_intersections_current_x;
    delete[] edge_intersections_current_y;
    delete[] edge_intersections_current_z;
}





}

#endif // _MARCHING_CUBES_H
