import cmlkit
import cmlkit.regression as cmlr
import cmlkit.indices as cmlki
import cmlkit.losses as cmll

# Load model spec and data
model = cmlkit.ModelSpec.from_yaml('model.spec.yml')
data = cmlkit.load_dataset('kaggle')

# Train/test split
train, test = cmlki.twoway_split(data.n, 400)

# Compute rmse, mae and r2 ... in one line!
loss = cmlr.idx_compute_loss(data, model, train, test, lossf=cmll.loss)

print(cmll.pretty_loss(loss))

# Note that the above computes the representations on the fly, and is therefore quite
# slow. It would be better to compute the MBTR ahead of time. 

# See compute_mbtr.py, and then run
# regression_with_explicit_mbtr.py, which makes use of it.
