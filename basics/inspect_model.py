import cmlkit

# Load model spec
spec = cmlkit.ModelSpec.from_yaml('model.spec.yml')

# Print will give us a nice overview of the model
print(spec)
