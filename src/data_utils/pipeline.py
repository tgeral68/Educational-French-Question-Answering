from collections.abc import Callable

class Pipeline():
    def __init__(self, *args):
        self.p_elements_list = list(args)

    def __add__(self, p_element):
        self.p_elements_list.append(p_element)
    
    def __sub__(self, p_element):
        self.p_elements_list.remove(p_element)
    
    def __call__(self, data):
        for p_el in self.p_element:
            data = p_el(data)