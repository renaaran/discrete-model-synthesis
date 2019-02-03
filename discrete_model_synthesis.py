#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:09:55 2019

Implementation of the Discrete Model Synthesis algorithm 3.4 from Paul Merrell's 
discrete model synthesis chapter of his dissertation. (2D)

@author: Renato Barros Arantes
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import choice

class DiscreteModelSynthesis(object):
    
    class Pattern(object):
        def __init__(self, pattern):
            self.pattern = np.copy(pattern)
            
        def __eq__(self, another):
            return np.array_equal(self.pattern, another.pattern)
        
        def __hash__(self):
            return hash(str(self.pattern))
        
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
        self.M = np.full((self.model_x_size, self.model_y_size), -1, dtype=int)
        
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
        self.pattern_matrix = np.zeros((pattern_matrix_cols, 
                                        pattern_matrix_rows), dtype=int)
        for x, y in itertools.product(range(0, X, block_size), 
                                      range(0, Y, block_size)):
            
            # allocate block for the new pattern
            block = np.full((block_size, block_size), -1, 
                            dtype=DiscreteModelSyntehsis.Pattern)
            
            for i, j in itertools.product(range(block_size), range(block_size)):
                block[i, j] = self.source_image.getpixel((x+i, y+j))
                
            pattern = DiscreteModelSyntehsis.Pattern(block)
            if pattern not in self.source_image_patterns:
                self.source_image_patterns[pattern] = current_pattern_id
                current_pattern_id += 1
                
            px = x//block_size
            py = y//block_size
            self.pattern_matrix[px, py] = self.source_image_patterns[pattern]

        self.patterns_ids = self.source_image_patterns.values()
        self.number_of_patterns = len(self.source_image_patterns)

    def calculate_probability_distribution(self):
        '''
            Calculates the labels probability distribution over patterns.
        '''
        patterns_count = {}
        for x, y in itertools.product(range(self.pattern_matrix.shape[0]), 
                                      range(self.pattern_matrix.shape[1])):
            pattern_id = self.pattern_matrix[x, y]
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

        for x, y in itertools.product(range(self.pattern_matrix.shape[0]-1), 
                                      range(self.pattern_matrix.shape[1]-1)):
            pattern = self.get_pattern(x, y)
            neighbour_right_pattern = self.get_pattern(x+1, y)
            neighbour_bellow_pattern = self.get_pattern(x, y+1)
            self.x_transition_matrix[pattern, neighbour_right_pattern]  = 1
            self.y_transition_matrix[pattern, neighbour_bellow_pattern] = 1
            
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
                            
    def get_pattern(self, x, y):
        '''
            Returns the pattern at the given coordinate in the mattern matrix.
            
            :param x: x pattern coordinate.
            :param y: y pattern coordinate.
            :return: returns the label at coordinate (x,y).
        '''
        return self.pattern_matrix[x, y]
    
    def update_neighbor(self, v, d, tm):
        '''
            Implements algorithm 3.3
        '''
        vd = (v[0]+d[0], v[1]+d[1])
        # outside border...
        if vd not in self.catalog:
            return
        # Check if each label c belongs in the catalog at v + d
        for c in range(self.number_of_patterns):
            if c in self.catalog[vd]:
                is_inconsistent = True
                # There is no b such that C[v, b] = 1 and T[b, c] = 1
                for b in range(self.number_of_patterns):
                    if b in self.catalog[v] and tm[b, c] == 1:
                        is_inconsistent = False
                        break
                if is_inconsistent:
                    self.patterns_to_update.append(vd)
                    self.catalog[vd].remove(c)
            
    def update_catalog(self, p):
        '''
            Update catalog to reflect assigning of label b to point p.
        '''
        x = p[0]
        y = p[1]
        b = self.M[x, y] # p label
        self.patterns_to_update = [p] # a stack of points to update.
        # Since label b is assigned to p, remove all other labels.
        self.catalog[p] = set([b])
        while len(self.patterns_to_update) > 0:
            v = self.patterns_to_update.pop()
            self.update_neighbor(v, (1, 0), self.x_transition_matrix) # i
            self.update_neighbor(v, (-1, 0), self.x_transition_matrix_transpose) # -i
            self.update_neighbor(v, (0, 1), self.y_transition_matrix) # j
            self.update_neighbor(v, (0, -1), self.y_transition_matrix_transpose) # -j
      
    def execute(self):
        '''
            Algorithm 3.1 Discrete Model Synthesis Algorithm
        '''
        self.catalog = {}
        # no label has been assigned to this point yet.
        for x, y in itertools.product(range(0, self.model_x_size), 
                                      range(0, self.model_y_size)):
            p = (x, y)
            self.catalog[p] = set(self.patterns_ids)
        ############
        self.M[0, 0] = self.pattern_matrix[0, 0]
        self.catalog[(0, 0)] = set([self.pattern_matrix[0, 0]])
        # Loop through each block
        for x, y in itertools.product(range(0, self.model_x_size), 
                                      range(0, self.model_y_size)):
                p = (x, y)
                # Check if the catalog is empty
                assert len(self.catalog[p]) != 0, 'Point = {}'.format(p)
                # Select any value of b for which C[p, b] = 1 at random.
                b = self.weighted_random_choice(self.catalog[p])
                self.M[x, y] = b
                self.update_catalog(p)

    def get_output_image(self):
        
        patterns = {}
        for pattern, pattern_id in self.source_image_patterns.items():
            patterns[pattern_id] = pattern
        output_image = Image.new('RGB', (self.output_x_size, self.output_y_size))
        pixels = output_image.load() # create the pixel map
        for x, y in itertools.product(range(self.model_x_size), 
                                      range(self.model_y_size)):
            pattern_id = self.M[x, y]
            pattern = patterns[pattern_id]
            px = x*self.pattern_block_size
            py = y*self.pattern_block_size
            for i in range(self.pattern_block_size):
                for j in range(self.pattern_block_size):
                    color = pattern.pattern[i,j]
                    if color != -1 and \
                        px+i < self.output_x_size and \
                        py+j < self.output_y_size:
                        pixels[px+i, py+j] = color
            
        plt.imshow(output_image)
        plt.show()
        return output_image
                                   
if __name__ == '__main__':
    sample_name = 'home-thumb.png'
    image = Image.open("samples/{}".format(sample_name))
    plt.imshow(image)
    plt.show()
    X = image.size[0]
    Y = image.size[1]
    print('x={}, y={}'.format(X, Y))
    for i in range(10):
        print(i)
        dms = DiscreteModelSynthesis(image, X, Y, pattern_block_size = 3)
        pm = dms.pattern_matrix
        tm = dms.x_transition_matrix
        dms.execute()
        dms.get_output_image()
