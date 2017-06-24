import api
import sys

def main(args):
    batch_size = 32  # Default
    if len(args) == 2:
        batch_size = int(args[1])
    if args[0] == "train":
        api.train(batch_size, True, True, True, 100)
    elif args[0] == "finetune":
        api.train(batch_size, False, True, True, 100)
    elif args[0] == "preprocess_questions":
        api.prepare_data()
    elif args[0] == "extract_training_features":
        api.extract_features(batch_size, "training")
    elif args[0] == "extract_validation_features":
        api.extract_features(batch_size, "validation")

if __file__ == "__main__":
    main(sys.argv[1:])


