import os
import re
import shutil
import tempfile
from glob import glob

from fire import Fire
from icecream import ic
from rich.progress import track
from sklearn.model_selection import train_test_split


def glob_images(rootdir):
    types = (ic(os.path.join(rootdir, '**', '*.'+tp)) for tp in ('png', 'PNG', 'gif', 'GIF', 'jpeg', 'jpg', 'JPG', 'JPEG')) # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend([os.path.abspath(f) for f in glob(files, recursive=True)])
    ic(len(files_grabbed))
    return files_grabbed

class Transform(object):
    """Organize data into 
    |->CLASS_1
    |   |->xxx.jpg
    |   |->ooo.jpg
    |->CLASS_2
    |->CLASS_3
    ...
    """
    def __init__(self, rootdir, copy=True, dry=False, transforms=[], pattern='', train_size=0.8, test_size=0.2, debug=True, out='', save_test=True, save_train=True):
        if not debug:
            ic.disable()
            
        self.dry = dry
        self.rootdir = ic(rootdir)
        self.transforms = ic(transforms) 
        self.pattern = pattern
        self.train_size = train_size
        self.test_size = test_size
        self.classes = []
        self.filenames = []
        self.out = ic(out) if out != '' else rootdir
        self.copy = copy
        if dry and out=='':
            self.out =  tempfile.mkdtemp()
        # self.split = split

    def run(self):
        try:
            for transform in self.transforms:
                eval(f'self.{transform}()')

            train_classes, test_classes, train_filenames, test_filenames = train_test_split(self.classes, self.filenames, train_size=self.train_size, test_size=self.test_size,stratify=self.classes, shuffle=True, random_state=42)
            self.save_train_data(classes=train_classes, filenames=train_filenames)
            self.save_test_data(classes=test_classes,  filenames=test_filenames)
        except Exception as e:
            ic(e)
            return 1
        if self.dry:
            try:
                self.test_out()
                ic("LOOKS GOOD!")
                shutil.rmtree(self.out)
                return 0
            except AssertionError as e:
                ic(e, 'Assertion Error')
                shutil.rmtree(self.out)
                return 1
            
    def cut_intermediate_dirs(self):
        root = self.rootdir
        files = glob_images(root)
        
        possible_classes = []
        basename = ic(os.path.basename(os.path.normpath(root)))
        for f in files: 
            fsplit = f.split(os.sep)
            for i, c in enumerate(fsplit[fsplit.index(basename): -1]):
                try:
                    possible_classes[i].add(c)
                except IndexError:
                    possible_classes.append(set([c]))
                    
        mx_classes = 0
        for i, pc in enumerate(ic(possible_classes)):
            if mx_classes <= len(pc):
                mx_classes = len(pc)                
                class_idx = fsplit.index(basename)+i            
                
        class_labels = ic(possible_classes[class_idx - fsplit.index(basename)])
        for i, f in enumerate(files):
            fsplit = f.split(os.sep)
            label = fsplit[class_idx]
            
            if label not in class_labels: # for those fucking datasets with weird structures!!!!!
                continue
            # f = os.path.join('/',*fsplit[:fsplit.index(basename)+1],label,fsplit[-1])
            self.filenames.append(ic(f))
            self.classes.append(ic(label))
            
            if i > 10:
                renable =True
                ic.disable()
        if renable:
            ic.enable()
            
    def dirname_to_class(self):
        files = glob_images(self.rootdir)
        for f in track(files, description='Extracting class from filename...'):
            c = os.path.basename(os.path.dirname(f))
            self.classes.append(c)
            self.filenames.append(f)

    def filename_to_dir(self):
        files = glob_images(self.rootdir)
        assert len(files) > 0, f'No files found. Check rootdir={self.rootdir}'
        for f in track(files, description='Extracting class from filename...'):
            try:
                c = re.search(self.pattern, f).groups()[0]                
                self.classes.append(c)
                self.filenames.append(f)
            except AttributeError as e:
                raise AttributeError('File without class found. Check regex pattern.')                
                # cls = match.groups()
    
    def save_test_data(self, filenames, classes):
        rootdir = self.out
        for c in set(classes):
            os.makedirs(os.path.join(rootdir, 'test', c), exist_ok=True)

        for c, f in zip(classes, filenames):
            if self.dry or self.copy:
                shutil.copy(f, os.path.join(rootdir, 'test', c, os.path.basename(f)))
            else:
                os.replace(f, os.path.join(rootdir, 'test', c, os.path.basename(f)))

    def save_train_data(self, filenames, classes):
        rootdir = self.out
        for c in set(classes):
            os.makedirs(os.path.join(rootdir, 'train', c), exist_ok=True)

        for c, f in zip(classes, filenames):
            if self.dry or self.copy:
                shutil.copy(f, os.path.join(rootdir, 'train', c, os.path.basename(f)))
            else:          
                os.replace(f, os.path.join(rootdir, 'train', c, os.path.basename(f)))

    def test_out(self):
        assert set(os.listdir(self.out)) == set(['test', 'train']), os.listdir(self.out)
        test_classes = sorted(os.listdir(os.path.join(self.out, 'test')))
        train_classes = sorted(os.listdir(os.path.join(self.out, 'train')))

        for testlabel, trainlabel in zip(test_classes, train_classes):
            assert testlabel==trainlabel, (testlabel, trainlabel)
            
            assert len(os.listdir(os.path.join(self.out, 'test', testlabel))) > 5, ''
            assert len(os.listdir(os.path.join(self.out, 'train', trainlabel))) > 30

    def test_classnames():
        pass

    def test_duplicates():
        pass


if __name__ == '__main__':
    Fire(Transform)
