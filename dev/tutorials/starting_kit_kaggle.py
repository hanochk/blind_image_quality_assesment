

# https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning
class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)


train_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )