from fire import Fire
from glob import glob
from rich import print
from rich.progress import track
import re
import os
from sklearn.model_selection import train_test_split

def glob_images(rootdir):
    types = (os.path.join(rootdir, '*.'+tp) for tp in ('png', 'PNG', 'gif', 'GIF', 'jpeg', 'jpg', 'JPG', 'JPEG')) # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob(files))
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
    def __init__(self, rootdir, transforms=[], pattern='', train_size=0.8, test_size=0.2):
        self.rootdir = rootdir
        self.transforms = transforms
        self.pattern = pattern
        self.train_size = train_size
        self.test_size = test_size
        self.classes = []
        self.filenames = []
        # self.split = split

    def run(self):
        for transform in self.transforms:
            eval(f'self.{transform}()')

        train_classes, test_classes, train_filenames, test_filenames = train_test_split(self.classes, self.filenames, train_size=self.train_size, test_size=self.test_size,stratify=self.classes, shuffle=True, random_state=42)
        self.save_train_data(classes=train_classes, filenames=train_filenames)
        self.save_test_data(classes=test_classes,  filenames=test_filenames)
        

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
        rootdir = self.rootdir
        for c in set(classes):
            os.makedirs(os.path.join(rootdir, 'test', c))

        for c, f in zip(classes, filenames):
            os.replace(f, os.path.join(rootdir, 'test', c, os.path.basename(f)))

    def save_train_data(self, filenames, classes):
        rootdir = self.rootdir
        for c in set(classes):
            os.makedirs(os.path.join(rootdir, 'train', c))

        for c, f in zip(classes, filenames):
            os.replace(f, os.path.join(rootdir, 'train', c, os.path.basename(f)))

    def dry(self):
        pass

    def test_classnames():
        pass

    def test_duplicates():
        pass


if __name__ == '__main__':
    Fire(Transform)