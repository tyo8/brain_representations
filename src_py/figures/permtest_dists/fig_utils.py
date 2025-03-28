import os
import pandas as pd


# takes list of lists (i.e., of form [[x1,x2,...,xi], [y1,y2,...,yj], ..., [z1, z2,...,zk]]) 
# and returns a nested list such that no subSETs are repeated (sublist order is ignored).
def unique_subls(nested_list):
    unique = set(tuple(sorted(a)) for a in nested_list)
    return list(unique)
