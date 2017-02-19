import tests_basis
from VQA.src.sentence_preprocess import preprocess

print(preprocess("hello word !"))
print("********************************")

print(preprocess("haven't you seen?"))
print("********************************")

print(preprocess("ok... I saw him !!"))
print("********************************")

print(preprocess("it's 5.30 o'clock, come fast"))
print("********************************")

print(preprocess("he ran from ahmed."))
print("********************************")

print(preprocess("does it contains sugar?"))
print("********************************")

print(preprocess("play football.."))  # problem here
print("********************************")

print(preprocess("ok"))
print("********************************")
