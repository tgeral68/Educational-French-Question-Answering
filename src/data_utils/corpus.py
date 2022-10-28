import os
import json
import tqdm



def to_hf(dataset, dataset_name, split_name, dataset_folder='$EFQADATA'):
    dataset_folder_path = os.path.expandvars(dataset_folder)
    os.makedirs(os.path.join(dataset_folder_path, dataset_name), exist_ok = True)
    with open(os.path.join(dataset_folder_path, dataset_name, split_name + '.jsonl'), "w") as split_file:
        for i, v in enumerate(tqdm.tqdm(dataset)):
            if(isinstance(v, list)):
                for d in v:
                    split_file.write(json.dumps({'hf_index': i,**d}) + '\n')
            else:
                split_file.write(json.dumps({'hf_index': i,**v}) + '\n')



class MixedDataset():
    def __init__(self, *kargs):
        self.datasets = [dataset for dataset in kargs]
        self.datasets_len = np.array([0] + [len(dataset) for dataset in self.datasets])
        self.datasets_len_cumsum = np.cumsum(self.datasets_len)
    
    def __len__(self):
        return np.sum(self.datasets_len)

    def __getitem__(self, index): 
        dataset_index = np.where(self.datasets_len_cumsum - int(index) > 0)[0][0] - 1
        return {**self.datasets[dataset_index][int(int(index) - self.datasets_len_cumsum[dataset_index])], 
            "index": index,
            "dataset_index": dataset_index}

    def jsonl_export(self, filepath, cache=100, add_null_fields=[]):
        with open(filepath, 'w') as f:
            cached_lines = ""
            for i, data in enumerate(self):
                
                for field in add_null_fields:
                    if field not in data:
                        data[field] = None
                for k, v in data.items():
                    if(isinstance(v, np.ndarray)):
                        data[k] = v.tolist()
                    if(isinstance(v, torch.Tensor)):
                        data[k] = v.tolist()
                if(i % cache == 0):
                    print(i)
                    f.write(cached_lines)
                    cached_lines = ""
                cached_lines += json.dumps(data)+'\n'
            f.write(cached_lines)

class UniformSamplerDataset():
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        cdata = self.dataset[index]
        return cdata[self.rng.integers(0, len(cdata)-1)]

class SelectDataset():
    def __init__(self, dataset, keys='answers'):
        self.dataset = dataset
        self.key = key
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        cdata = self.dataset[index]
        return cdata[self.key]