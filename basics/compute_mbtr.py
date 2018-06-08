import cmlkit
import cmlkit.regression as cmlr
import cmlkit.indices as cmlki
import cmlkit.losses as cmll

# Load model spec and data
spec = cmlkit.ModelSpec.from_yaml('model.spec.yml')

# We assume that you have run the 'dataset' tutorial, and
# have the 'kaggle' dataset somewhere in your CML_DATASET_PATH
data = cmlkit.load_dataset('kaggle')

mbtr = cmlkit.MBTR(data, spec)
mbtr.save()
