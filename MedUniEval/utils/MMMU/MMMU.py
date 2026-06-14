
from .eval_val import eval_MMMU_val
from .eval_test import eval_MMMU_test

class MMMU:
    def __init__(self,model,dataset_path,output_path,split = "test",subset = "Medical"):
        self.split = split
        self.subset = subset
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "MMMU/MMMU"
        self.samples = []

    def load_data(self):
        pass

    def eval(self):
        model = self.model
        dataset_path = self.dataset_path
        output_path = self.output_path

        if self.subset == "Medical":
            subset = "Health and Medicine"
        elif self.subset == "Science":
            subset = "Science"
        if self.split == "test":
            matrics = eval_MMMU_test(model,dataset_path,output_path,subset)
        else:
            matrics = eval_MMMU_val(model,dataset_path,output_path,subset)
        return matrics

                