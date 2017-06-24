import api
import sys

def main(args):
    batch_size = 32  # Default
    if len(args) == 2:
        batch_size = int(args[1])
        
    if args[0] == "train":
        print("Training from scratch ...")
        print("Batch size is ", batch_size)
        api.train(batch_size, True, True, True, 100)
    elif args[0] == "finetune":
        print("Finetune checkpoint ...")
        print("Batch size is ", batch_size)
        api.train(batch_size, False, True, True, 100)
    elif args[0] == "preprocess_questions":
        print("Preprocess Questions ...")
        api.prepare_data()
    elif args[0] == "extract_training_features":
        print("Extract Training Images Features ...")
        print("Batch size is ", batch_size)
        api.extract_features(batch_size, "training")
    elif args[0] == "extract_validation_features":
        print("Extract Validation Images Features ...")
        print("Batch size is ", batch_size)
        api.extract_features(batch_size, "validation")


if __name__ == "__main__":
    main(sys.argv[1:])


