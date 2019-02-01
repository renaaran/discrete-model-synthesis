#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:09:55 2019

Implementation of the Discrete Model Synthesis algorithm 3.4 from Paul Merrell's 
discrete model synthesis chapter of his dissertation. (2D)

@author: Renato Barros Arantes
"""

import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DiscreteModeSyntehsis(object):
    
    def __init__(self, source_image, 
                 output_x_size, output_y_size,
                 block_x_size = 10, block_y_size = 10):
        self.source_image = source_image.convert('RGB')
        self.colors_set = set()
        self.colors_to_label = {}
        self.catalog = {}
        self.labels = []
        self.points_to_update = []
        self.X = source_image.size[0]
        self.Y = source_image.size[1]
        self.label_image = np.zeros((self.X, self.Y), dtype=int)
        self._count_number_of_colors()
        self.number_of_colors = len(self.colors_set)
        print(self.number_of_colors)
        self._calc_transition_matrix()
        self.block_x_size = block_x_size
        assert self.block_x_size > 0 and self.block_x_size <= self.X
        self.block_y_size = block_y_size
        assert self.block_y_size > 0 and self.block_y_size <= self.Y
        # create initial solution
        self.output_x_size = output_x_size
        self.output_y_size = output_y_size
        self.M = np.full((output_x_size, output_y_size), -1, dtype=int)
            
    def _count_number_of_colors(self):
        '''
            Count the number of distinct color, or labels, in the give image.
            
            :param image: source image or example model (E)
            :return: returns nothing
         '''
        for x, y in itertools.product(range(self.X), range(self.Y)):
            color = self.source_image.getpixel((x, y))
            self.colors_set.add(color)
        print('Number of distinct colors = {}'.format(len(self.colors_set)))
        # creates a unique id colors map 
        for idx, color in enumerate(self.colors_set):
            self.colors_to_label[color] = idx
        # list of possibles labels
        self.labels = self.colors_to_label.values()
        # getpixel is very slow, so now we create a matrix with the same shape
        # as the original image but now with the labels instead of colors.
        for x, y in itertools.product(range(self.X), range(self.Y)):
            color = self.source_image.getpixel((x, y))
            self.label_image[x, y] = self.colors_to_label[color]
        
        plt.imshow(self.source_image)
        plt.show()
#        plt.matshow(self.label_image)
#        plt.show()
        
    def _get_label(self, x, y):
        '''
            Returns the label, or color index, of the give point.
            
            :param x: x point coordinate.
            :param y: y point coordinate.
            :return: returns the label at coordinate (x,y).
        '''
        return self.label_image[x, y]
    
    def _calc_transition_matrix(self):
        '''
            For a given input image (E), the set of equations bellow acts as a constraint 
            on the output image (M) and is called the 'adjacency constraint':
                
                M(X) = E(X')
                M(X + D) = E(X' + D)
                
            Where D is one of the unit vectos {i,j}.
                
            This function calculates the transition matrix that translates the
            adjacency constraint above.
            
            :param image: source image or example model (E)
            :return: returns the transition matrix for the source model.
        '''    
        self.x_transition_matrix = np.zeros((self.number_of_colors, 
                                           self.number_of_colors), dtype=bool)
        self.y_transition_matrix = np.zeros((self.number_of_colors, 
                                           self.number_of_colors), dtype=bool)

        for x, y in itertools.product(range(self.X-1), range(self.Y-1)):
            point_color = self._get_label(x, y)
            neighbour_right_color = self._get_label(x+1, y)
            neighbour_bellow_color = self._get_label(x, y+1)
            self.x_transition_matrix[point_color, neighbour_right_color]  = 1
            self.y_transition_matrix[point_color, neighbour_bellow_color] = 1
            
        self.x_transition_matrix_transpose = self.x_transition_matrix.transpose()
        self.y_transition_matrix_transpose = self.y_transition_matrix.transpose()
            
    def _inside_block(self, p, q):
        '''
            :param p,q: Two points p and q
            :return: True if p is inside the block with a corner at the point 
            ((qx*mx)/2, (qy*my)/2)
        '''
        l = lambda x, m, d: (x*m)//2+d
        px = p[0]
        py = p[1]
        qx = q[0]
        qy = q[1]
        mx = self.block_x_size
        my = self.block_y_size
        return l(qx, mx, 0) <= px < l(qx, mx, mx) and \
               l(qy, my, 0) <= py < l(qy, my, my)
      
    def _update_neighbor(self, v, d, tm):
        '''
            Implements algorithm 3.3
        '''
        vd = (v[0]+d[0], v[1]+d[1])
        # outside border...
        if vd not in self.catalog:
            return
        # Check if each label c belongs in the catalog at v + d
        for c in range(self.number_of_colors):
            if c in self.catalog[vd]:
                is_inconsistent = True
                # There is no b such that C[v, b] = 1 and T[b, c] = 1
                for b in range(self.number_of_colors):
                    if b in self.catalog[v] and tm[b, c] == 1:
                        is_inconsistent = False
                        break
                if is_inconsistent:
                    self.points_to_update.append(vd)
                    self.catalog[vd].remove(c)
            
    def _update_catalog(self, p):
        '''
            Update catalog to reflect assigning of label b to point p.
        '''
        x = p[0]
        y = p[1]
        b = self.M[x, y] # p label
        self.points_to_update.append(p) # u is a stack of points to update.
        # Since label b is assigned to p, remove all other labels.
        self.catalog[p] = set([b])
        while len(self.points_to_update) > 0:
            v = self.points_to_update.pop()
            self._update_neighbor(v, (1, 0), self.x_transition_matrix) # i
            self._update_neighbor(v, (-1, 0), self.x_transition_matrix_transpose) # -i
            self._update_neighbor(v, (0, 1), self.y_transition_matrix) # j
            self._update_neighbor(v, (0, -1), self.y_transition_matrix_transpose) # -j
        
    def get_output_image(self):
        label_to_color = list(self.colors_set)
        output_image = Image.new('RGB', (self.output_x_size, self.output_y_size))
        pixels = output_image.load() # create the pixel map
        for x, y in itertools.product(range(self.output_x_size), 
                                      range(self.output_y_size)):
            pixels[x, y] = label_to_color[self.M[x, y]]
        plt.imshow(output_image)
        plt.show()
        return output_image
        
    def execute3_4(self):
        '''
            Algorithm 3.4 Final Discrete Model Synthesis Algorithm
        '''
        # Loop through each block
        print(2*self.output_x_size//self.block_x_size, 2*self.output_y_size//self.block_y_size)
        for qx, qy in itertools.product(
                range(2*self.output_x_size//self.block_x_size), 
                range(2*self.output_y_size//self.block_y_size)): 
            q = (qx, qy)
            print(q)
            self.get_output_image()
            # no label has been assigned to this point yet.
            for x, y in itertools.product(range(self.output_x_size), 
                                          range(self.output_y_size)):
                p = (x, y)
                self.catalog[p] = set(self.labels)
            # Save the current value of M
            M0 = self.M.copy()
            # Include all assignments in the catalog
            for x, y in itertools.product(range(self.output_x_size), 
                                          range(self.output_y_size)):
                p = (x, y)
                # Add everything outside the current block.
                if not self._inside_block(p, q): 
                    self._update_catalog(p)
            # Run Algorithm 3.1 within the block.
            failed = False
            for x, y in itertools.product(range(self.output_x_size), 
                                          range(self.output_y_size)):
                p = (x, y)
                if self._inside_block(p, q): 
                    # Check if the catalog is empty
                    if len(self.catalog[p]) == 0:
                        failed = True
                        print('Failed!!!:(')
                        break
                    else:
                        # Select any value of b for which C[p, b] = 1 at random.
                        b = random.choice(tuple(self.catalog[p]))
                        self.M[x, y] = b
                        self._update_catalog(p)
            # If M becomes inconsistent, restore its previous value.
            if failed:
                self.M = M0.copy() 

    def execute3_1(self):
        '''
            Algorithm 3.1 Discrete Model Synthesis Algorithm
        '''
        # no label has been assigned to this point yet.
        for x, y in itertools.product(range(self.output_x_size), 
                                      range(self.output_y_size)):
            p = (x, y)
            self.catalog[p] = set(self.labels)
        # Loop through each block
        print(self.output_x_size, self.output_y_size)
        for x in range(self.output_x_size):
            #self.get_output_image()
            for y in range(self.output_y_size): 
                p = (x, y)
                # Check if the catalog is empty
                assert len(self.catalog[p]) != 0
                # Select any value of b for which C[p, b] = 1 at random.
                b = random.choice(tuple(self.catalog[p]))
                self.M[x, y] = b
                self._update_catalog(p)
                           
if __name__ == '__main__':
#    random.seed(42)
    sample_name = 'home-thumb.png'
    image = Image.open("samples/{}".format(sample_name))
    X = image.size[0]
    Y = image.size[1]
    print('x={}, y={}'.format(X, Y))
    dms = DiscreteModeSyntehsis(image, X, Y)
    dms.execute3_1()
    dms.get_output_image()
    mx = dms.x_transition_matrix