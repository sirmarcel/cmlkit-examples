import cmlkit
import cmlkit.regression as cmlr
import cmlkit.indices as cmlki
import cmlkit.losses as cmll

# Load model spec and data
spec = cmlkit.ModelSpec.from_yaml('model.spec.yml')
data = cmlkit.load_dataset('kaggle')
mbtr = cmlkit.MBTR.from_file('kaggle_model.mbtr.npy')

# Train/test split
train, test = cmlki.twoway_split(data.n, 400)

# Compute rmse, mae and r2 ... in one line!
loss = cmlr.idx_compute_loss(data, spec, train, test, lossf=cmll.loss, rep=mbtr)
print('Loss for fecreal:')
print(cmll.pretty_loss(loss) + '\n')

# Note that the spec defines a model predicting a quantity that we might not be interested in:
# fecreal, the formation energy per unit cell. What happens if we want the loss
# on fe, the original target quantity for the kaggle competition? Easy:

loss = cmlr.idx_compute_loss(data, spec, train, test, lossf=cmll.loss, rep=mbtr, target_property='fe')
print('Loss for fe:')
print(cmll.pretty_loss(loss) + '\n')

# Now, the following is happening: The model is trained on the property fecreal, and predicts this quantity,
# but the predictions are automatically converted to fe. Neat!

# Sidenote: property conversion is currently implemented in a *very* hacky way (see the property_converter.py in cmlkit),
# but this will improve. Currently, the only way to write your own conversions is to fork cmlkit directly.

# Let's also look at the log loss.
loss = cmlr.idx_compute_loss(data, spec, train, test, lossf=cmll.logloss, rep=mbtr, target_property='fe')
print('Log Loss for fe:')
print(cmll.pretty_loss(loss) + '\n')
