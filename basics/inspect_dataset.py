import cmlkit

# We assume that you have run the 'dataset' tutorial, and
# have the 'kaggle' dataset somewhere in your CML_DATASET_PATH
data = cmlkit.load_dataset('kaggle')

print(data)

# This will print an in-depth report of the dataset,
# and give us good suggestions on how to set the MBTR