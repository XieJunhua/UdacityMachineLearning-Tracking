import numpy as np
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()

def test_one_hot_encode(one_hot_encode):
  test_shape = np.random.choice(range(1000))
  test_numbers = np.random.choice(range(10), test_shape)
  one_hot_out = one_hot_encode(test_numbers)

  assert type(one_hot_out).__module__ == np.__name__, \
    'Not Numpy Object'

  assert one_hot_out.shape == (test_shape, 10), \
    'Incorrect Shape. {} shape found'.format(one_hot_out.shape)

  n_encode_tests = 5
  test_pairs = list(zip(test_numbers, one_hot_out))
  test_indices = np.random.choice(len(test_numbers), n_encode_tests)
  labels = [test_pairs[test_i][0] for test_i in test_indices]
  enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
  new_enc_labels = one_hot_encode(labels)

  assert np.array_equal(enc_labels, new_enc_labels), \
    'Encodings returned different results for the same numbers.\n' \
    'For the first call it returned:\n' \
    '{}\n' \
    'For the second call it returned\n' \
    '{}\n' \
    'Make sure you save the map of labels to encodings outside of the function.'.format(enc_labels, new_enc_labels)

def one_hot_encode(x):
  ohe.

