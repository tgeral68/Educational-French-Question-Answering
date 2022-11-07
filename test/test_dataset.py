import unittest
import torch

from src.data_utils import corpus, pb_corpus
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class TestDataset(unittest.TestCase):
    
    def setUp(self):
        self.data = [{
            "title": "De l\u2019Empire d\u2019Autriche \u00e0 l\u2019Autriche\u2011Hongrie",
            "qas": [
                {
                    "question": "Quel est la cons\u00e9quence de l'\u00e9chec de la r\u00e9volution populaire Hongroise.",
                    "answers": [
                        [
                            {
                                "text": ", l\u2019Empire d\u2019Autriche et sa dynastie, les Habsbourg, sont restaur\u00e9s dans toute leur puissance",
                                "answer_start": 80,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "long_answers": [
                        [
                            {
                                "text": ", l\u2019Empire d\u2019Autriche et sa dynastie, les Habsbourg, sont restaur\u00e9s dans toute leur puissance. Cependant, la mont\u00e9e des sentiments nationaux et la d\u00e9faite face \u00e0 la Prusse \u00e0 Sadowa (1866) obligent Fran\u00e7ois\u2011Joseph\u00a0Ier, l\u2019empereur, \u00e0 faire \u00e9voluer cet empire multinational, v\u00e9ritable mosa\u00efque de peuples et de langues. Le pouvoir imp\u00e9rial tente alors de r\u00e9pondre \u00e0 certaines demandes, notamment en accordant une autonomie politique et administrative \u00e0 la Hongrie, tout en maintenant l\u2019unit\u00e9 de l\u2019Empire.",
                                "answer_start": 80,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "question_type": "knowledge"
                },
                {
                    "question": "En quelle ann\u00e9e la Prusse est d\u00e9faite \u00e0 Sadowa ?",
                    "answers": [
                        [
                            {
                                "text": "1866",
                                "answer_start": 262,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "long_answers": [
                        [
                            {
                                "text": "Cependant, la mont\u00e9e des sentiments nationaux et la d\u00e9faite face \u00e0 la Prusse \u00e0 Sadowa (1866) obligent Fran\u00e7ois\u2011Joseph\u00a0Ier, l\u2019empereur, \u00e0 faire \u00e9voluer cet empire multinational",
                                "answer_start": 175,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "question_type": "factual"
                },
                {
                    "question": "D'apr\u00e8s l'extrait de \"L'allemagne depuis la guerre de 1866\" d'Emile de Laveleye, comment r\u00e9sumeriez-vous son point de vue sur l'unit\u00e9 de l'Autriche ?",
                    "answers": [
                        [
                            {
                                "text": "Chacun est attach\u00e9 \u00e0 sa province, nul ne l\u2019est \u00e0 l\u2019Empire",
                                "answer_start": 605,
                                "paragraph_id": 1
                            }
                        ]
                    ],
                    "long_answers": [
                        [
                            {
                                "text": "un propri\u00e9taire me disait qu\u2019il avait absolument besoin de conna\u00eetre cinq langues\u00a0: le latin pour les anciennes pi\u00e8ces officielles, l\u2019allemand pour ses relations avec Vienne, le hongrois pour prendre la parole dans la di\u00e8te, enfin le valaque et le serbe pour donner des ordres \u00e0 ses ouvriers. ",
                                "answer_start": 119,
                                "paragraph_id": 1
                            }
                        ]
                    ],
                    "question_type": "knowledge"
                },
                {
                    "question": "En quelle ann\u00e9e a eu lieu la d\u00e9faite de la Hongrie face \u00e0 la Prusse \u00e0 Sadowa ?",
                    "answers": [
                        [
                            {
                                "text": "1866",
                                "answer_start": 262,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "long_answers": [
                        []
                    ],
                    "question_type": "factual"
                },
                {
                    "question": "Comment s'appelle la dynastie de l'Empire d'Autriche ?",
                    "answers": [
                        [
                            {
                                "text": "les Habsbourg",
                                "answer_start": 118,
                                "paragraph_id": 0
                            }
                        ]
                    ],
                    "long_answers": [
                        []
                    ],
                    "question_type": "knowledge"
                }
            ],
            "paragraphs": [
                {
                    "title": "",
                    "content": "Apr\u00e8s l\u2019\u00e9chec de la r\u00e9volution populaire hongroise lors du Printemps des peuples, l\u2019Empire d\u2019Autriche et sa dynastie, les Habsbourg, sont restaur\u00e9s dans toute leur puissance. Cependant, la mont\u00e9e des sentiments nationaux et la d\u00e9faite face \u00e0 la Prusse \u00e0 Sadowa (1866) obligent Fran\u00e7ois\u2011Joseph\u00a0Ier, l\u2019empereur, \u00e0 faire \u00e9voluer cet empire multinational, v\u00e9ritable mosa\u00efque de peuples et de langues. Le pouvoir imp\u00e9rial tente alors de r\u00e9pondre \u00e0 certaines demandes, notamment en accordant une autonomie politique et administrative \u00e0 la Hongrie, tout en maintenant l\u2019unit\u00e9 de l\u2019Empire."
                },
                {
                    "title": "",
                    "content": "On pr\u00e9tend qu\u2019on trouve en Autriche vingt nationalit\u00e9s diff\u00e9rentes et dix\u2011huit idiomes. [\u2026] Aux environs de Temeswar1, un propri\u00e9taire me disait qu\u2019il avait absolument besoin de conna\u00eetre cinq langues\u00a0: le latin pour les anciennes pi\u00e8ces officielles, l\u2019allemand pour ses relations avec Vienne, le hongrois pour prendre la parole dans la di\u00e8te, enfin le valaque et le serbe pour donner des ordres \u00e0 ses ouvriers. [\u2026] L\u2019Autriche forme un assemblage bariol\u00e9 de groupes ethnographiques qui ne se sont pas m\u00eal\u00e9s, comme en France, de fa\u00e7on \u00e0 constituer une seule nation ayant le sentiment d\u2019une patrie commune. Chacun est attach\u00e9 \u00e0 sa province, nul ne l\u2019est \u00e0 l\u2019Empire. Vous trouvez des Hongrois, des Croates, des Tch\u00e8ques acharn\u00e9s, mais pas d\u2019Autrichiens.\n\n  \u00c9mile de Laveleye, \u00ab\u00a0L\u2019Allemagne depuis la guerre de 1866\u00a0\u00bb, article paru dans la Revue des Deux Mondes, 1868."
                }
            ]
        }]
        self.dataset = pb_corpus.FQAGPBDataset(
                self.data,
                sampler = lambda x : [x[0]],
                input_lang = "fr", output_lang = "fr"
            )
    def tearDown(self):
        pass

    def test_prepositional_dataset(self):
        pd = corpus.PrepositionalTokenDataset(self.dataset, question="[question_generation]")
        assert(pd[0]['question'] == "[question_generation]"+self.dataset[0]['question'])
    
    def test_tokenizer_mbart(self):
        pd = corpus.PrepositionalTokenDataset(self.dataset, question="[question_generation]")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer.add_tokens(['<hl>'] + ["[question_generation]"], special_tokens=True)

        tokenized_data = self.tokenizer(pd[0]['question'])['input_ids']
        detokenized_data_keep = self.tokenizer.decode(tokenized_data, skip_special_tokens=False)
        detokenized_data_skip = self.tokenizer.decode(tokenized_data, skip_special_tokens=True)
        assert(detokenized_data_skip == 'Quel est la conséquence de l\'échec de la révolution populaire Hongroise.')
        assert(detokenized_data_keep == 'en_XX[question_generation] Quel est la conséquence de l\'échec de la révolution populaire Hongroise.</s>')
