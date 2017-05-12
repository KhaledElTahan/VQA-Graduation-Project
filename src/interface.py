from model import evaluate
from sentence_preprocess import sentence2vecs
from f_extractor import get_features
from annotation_fetcher import get_top_answers

def answer_question_on_image(image, question):
	image_features = get_features(image)
	question_features = sentence2vecs(question)
	evaluation_logits = evaluate(image_features, question_features)
	answer_index = evaluation_logits.index(max(evaluation_logits))
	top_answers = get_top_answers()
	return top_answers[index]