from torch.distributions import Categorical
from scipy.stats import entropy

a = [0.1, 0.2, 0.3, 0.3, 0.1]
b = [0.4, 0.3, 0.1, 0.1, 0.1]

print(entropy(a, b))
print(Categorical)