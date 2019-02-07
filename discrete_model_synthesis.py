#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:09:55 2019

Implementation of the Discrete Model Synthesis algorithm 3.4 from Paul Merrell's 
discrete model synthesis chapter of his dissertation. (2D)

@author: Renato Barros Arantes
"""
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy.random import choice

class DiscreteModelSynthesis(object):
    
    class Pattern(object):
        def __init__(self, pixels):
            self.pixels = np.copy(pixels)
            
        def __eq__(self, another):
            return np.array_equal(self.pixels, another.pixels)
        
        def __hash__(self):
            return hash(str(self.pixels))
        
    def __init__(self, source_image, output_x_size, output_y_size, 
                 pattern_block_size = 2):
        self.source_image = source_image.convert('RGB')
        self.translate_source_image_to_pattern_matrix(pattern_block_size)
        self.calculate_probability_distribution()
        self.calculate_transition_matrix()
        self.pattern_block_size = pattern_block_size
        self.output_x_size = output_x_size
        self.output_y_size = output_y_size
        self.model_x_size = (output_x_size//pattern_block_size)*pattern_block_size
        self.model_y_size = (output_y_size//pattern_block_size)*pattern_block_size
        print(self.model_x_size, self.model_y_size)
        self.M = np.zeros((self.model_y_size, self.model_x_size), dtype=int)
        
    def translate_source_image_to_pattern_matrix(self, block_size):
        '''
            Creates a set o patterns. Each pattern is a square block taken upon
            the source image.
        '''
        # Creates the set of pattern within the source image.
        current_pattern_id = 0
        self.source_image_patterns = {}
        X = (self.source_image.size[0]//block_size)*block_size
        Y = (self.source_image.size[1]//block_size)*block_size
        pattern_matrix_cols = X//block_size
        pattern_matrix_rows = Y//block_size
        self.pattern_matrix = np.zeros((pattern_matrix_rows,
                                        pattern_matrix_cols), dtype=int)
        source_image_pixels = self.source_image.load()
        for y, x in itertools.product(range(0, Y, block_size),
                                      range(0, X, block_size)):
            
            # allocate block for the new pattern
            pixels = np.zeros((block_size, block_size, 3), dtype=int)
            for i, j in itertools.product(range(block_size), range(block_size)):
                p = source_image_pixels[x+j, y+i]
                pixels[i, j, 0] = p[0]
                pixels[i, j, 1] = p[1]
                pixels[i, j, 2] = p[2]

            pattern = DiscreteModelSynthesis.Pattern(pixels)
            if pattern not in self.source_image_patterns:
                self.source_image_patterns[pattern] = current_pattern_id
                current_pattern_id += 1
                
            row = y//block_size
            col = x//block_size
            self.pattern_matrix[row, col] = self.source_image_patterns[pattern]

        self.patterns_ids = set(self.source_image_patterns.values())
        self.number_of_patterns = len(self.source_image_patterns)
        print('number_of_patterns={}'.format(self.number_of_patterns))

    def calculate_probability_distribution(self):
        '''
            Calculates the labels probability distribution over patterns.
        '''
        patterns_count = {}
        for row, col in itertools.product(range(self.pattern_matrix.shape[0]),
                                          range(self.pattern_matrix.shape[1])):
            pattern_id = self.pattern_matrix[row, col]
            # count how many times each pattern appear.
            if not pattern_id in patterns_count:
                patterns_count[pattern_id] = 1
            else:
                patterns_count[pattern_id] += 1

        # calculate the probability distribution of each label
        self.probability_distribution = {}
        pm_elements_count = self.pattern_matrix.shape[0]*self.pattern_matrix.shape[1]
        for pattern_id, count in patterns_count.items():
            self.probability_distribution[pattern_id] = count/pm_elements_count

    def get_pattern(self, row, col):
        '''
            Returns the pattern at the given coordinate in the mattern matrix.
            
            :param row: row pattern coordinate.
            :param col: col pattern coordinate.
            :return: returns the label at coordinate (row,col).
        '''
        return self.pattern_matrix[row, col]
    
    def calculate_transition_matrix(self):
        '''
            For a given input image (E), the set of equations bellow acts as a constraint 
            on the output image (M) and is called the 'adjacency constraint':
                
                M(X) = E(X')
                M(X + D) = E(X' + D)
                
            Where D is one of the unit vectos {i,j}.
                
            This function calculates the transition matrix that translates the
            adjacency constraint above.
        ''' 
        
        self.x_transition_matrix = np.zeros((self.number_of_patterns, 
                                             self.number_of_patterns), dtype=bool)
        self.y_transition_matrix = np.zeros((self.number_of_patterns, 
                                             self.number_of_patterns), dtype=bool)
        Y = self.pattern_matrix.shape[0]
        X = self.pattern_matrix.shape[1]
        for row, col in itertools.product(range(Y), range(X)):
            pattern = self.get_pattern(row, col)
            if col+1 < X:
                neighbour_right_pattern = self.get_pattern(row, col+1)
                self.x_transition_matrix[pattern, neighbour_right_pattern] = 1
#            else:
#                # x loop! :)
#                row_beginning = self.get_pattern(row, 0)
#                self.x_transition_matrix[pattern, row_beginning] = 1
                
            if row+1 < Y:
                neighbour_bellow_pattern = self.get_pattern(row+1, col)
                self.y_transition_matrix[pattern, neighbour_bellow_pattern] = 1
#            else:
#               # y loop! :)
#                col_beginning = self.get_pattern(0, col)
#                self.y_transition_matrix[pattern, col_beginning] = 1
              
        self.x_transition_matrix_transpose = self.x_transition_matrix.transpose()
        self.y_transition_matrix_transpose = self.y_transition_matrix.transpose()

    def weighted_random_choice(self, candidates):
        list_of_candidates = []
        list_of_prob_dist = []
        for pattern_id in candidates:
            prob_dist = self.probability_distribution[pattern_id]
            list_of_candidates.append(pattern_id)
            list_of_prob_dist.append(prob_dist)
        array_of_prob_dist = np.array(list_of_prob_dist)
        array_of_prob_dist /= array_of_prob_dist.sum()
        return choice(list_of_candidates, 1, p=array_of_prob_dist)
       
    def exist_transition_to(self, transition_matrix, catalog, c):
         for b in catalog:
                if transition_matrix[b, c] == True:
                    return True
         return False
                     
    def update_neighbor(self, v, d, tm):
        '''
            The catalog C contains a list of acceptable labels at each point. 
            The label c is only acceptable at v + d, if there exists a label b 
            that is acceptable at point v meaning C(x, b) = 1 and that can be 
            adjacent to c meaning tm[b, c] = 1.
        '''
        vd = (v[0]+d[0], v[1]+d[1])
        # outside border...
        if vd not in self.catalog: return
        # Check if each label c belongs in the catalog at v + d
        vd_catalog = self.catalog[vd].copy()
        for c in vd_catalog:
            if not self.exist_transition_to(tm, self.catalog[v], c):
                # There is no b such that C[v, b] = 1 and T[b, c] = 1
                self.patterns_to_update.add(vd)
                self.catalog[vd].remove(c)
                assert len(self.catalog[vd]) != 0, \
                    'No transition from v={} to vd={}:  v_catalog={} -> vd_catalog={}' \
                    .format(v, vd, self.catalog[v], vd_catalog)
    
    def update_catalog(self, p):
        '''
            Update catalog to reflect assigning of label b to point p.
        '''
        y = p[0]
        x = p[1]
        # p label
        b = self.M[y, x] 
        # Since label b is assigned to p, remove all other labels.
        self.catalog[p] = set([b])
        # A stack of points to update.
        self.patterns_to_update = set([p]) 
        while len(self.patterns_to_update) > 0:
            v = self.patterns_to_update.pop()
            self.update_neighbor(v, (0, 1), self.x_transition_matrix) # i
            self.update_neighbor(v, (0, -1), self.x_transition_matrix_transpose) # -i
            self.update_neighbor(v, (1, 0), self.y_transition_matrix) # j
            self.update_neighbor(v, (-1, 0), self.y_transition_matrix_transpose) # -j
      
    def execute(self):
        '''
            Algorithm 3.1 Discrete Model Synthesis Algorithm
        '''
        self.catalog = {}
        # no label has been assigned to this point yet.
        for y, x in itertools.product(range(self.model_y_size),
                                      range(self.model_x_size)):
            p = (y, x)
            self.catalog[p] = self.patterns_ids.copy()
        # find a model M compatible with E
        for y, x in itertools.product(range(self.model_y_size),
                                      range(self.model_x_size)):
            p = (y, x)
            print(p)
            # Check if the catalog is empty
            assert len(self.catalog[p]) != 0, 'Point = {}'.format(p)
            # Select any value of b for which C[p, b] = 1 at random.
            b = self.weighted_random_choice(self.catalog[p])
            self.M[y, x] = b
            self.update_catalog(p)

    def get_output_image(self):
        
        patterns = {}
        for pattern, pattern_id in self.source_image_patterns.items():
            patterns[pattern_id] = pattern
        output_image = Image.new('RGB', (self.output_x_size, self.output_y_size))
        pixels = output_image.load() # create the pixel map
        for y, x in itertools.product(range(self.model_y_size),
                                      range(self.model_x_size)):
            pattern_id = self.M[y, x]
            pattern = patterns[pattern_id]
            px = x*self.pattern_block_size
            py = y*self.pattern_block_size
            for i, j in itertools.product(range(self.pattern_block_size),
                                          range(self.pattern_block_size)):
                color = pattern.pixels[i,j]
                if px+j < self.output_x_size and \
                   py+i < self.output_y_size:
                    pixels[px+j, py+i] = (color[0], color[1], color[2])
        
        plt.imshow(output_image)
        plt.show()
        return output_image
 
    def plot_patterns(self):
        patterns = {}
        for pattern, pattern_id in self.source_image_patterns.items():
            patterns[pattern_id] = pattern
            
        number_of_patterns = len(patterns)
        f, axarr = plt.subplots(1, number_of_patterns-1, figsize=(30,30))
        for i in range(1, number_of_patterns):
            axarr[i-1].imshow(patterns[i].pixels)
            axarr[i-1].axis('off')
            axarr[i-1].set_title(str(i))
        plt.show()
        
if __name__ == '__main__':
    np.random.seed(42)
    sample_name = 'Flowers'
    image = cv2.imread(Image.open("samples/{}.png".format(sample_name)))
    plt.imshow(image)
    plt.show()
    X = image.size[0]
    Y = image.size[1]
    print('Input image size: x={}, y={}'.format(X, Y))
    dms = DiscreteModelSynthesis(image, X, Y, pattern_block_size=3)
    tmx = dms.x_transition_matrix
    tmy = dms.y_transition_matrix
    tm = dms.pattern_matrix
    for i in range(1):
        print(i)
        dms.execute()
        dms.get_output_image()
