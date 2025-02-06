import pandas as pd
import os

class CEA_Evaluator:
    def __init__(self, answer_file_path, round=1):
        """
    `round` : Holds the round for which the evaluation is being done.
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
        self.answer_file_path = answer_file_path
        self.round = round

    def _evaluate(self, client_payload, _context={}):
        """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

        gt_cell_ent = dict()
        # gt = pd.read_csv(self.answer_file_path, encoding='utf-8', errors='ignore', delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
        #                  dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        with open(self.answer_file_path, mode='r', encoding='utf-8', errors='ignore') as ffff:
            gt = pd.read_csv(ffff, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                         dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        
        for index, row in gt.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
            gt_cell_ent[cell] = row['entity']

        false_cells, correct_cells, annotated_cells = set(), set(), set() # 错误注释的单元格，正确注释的单元格，注释的单元格
        # sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
        #                   dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        with open(submission_file_path, mode='r', encoding='utf-8', errors='ignore') as fff:
            sub = pd.read_csv(fff, delimiter=',', names=['tab_id', 'row_id', 'col_id', 'entity'],
                          dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
        
        for index, row in sub.iterrows():
            cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
            if cell in gt_cell_ent:
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = row['entity']
                if not annotation.lower() == 'nil' and not annotation.startswith('http://dbpedia.org/resource/'):
                    annotation = 'http://dbpedia.org/resource/' + annotation

                if annotation.lower() in gt_cell_ent[cell].lower().split():
                    correct_cells.add(cell)
                else:
                    false_cells.add(cell)
                    # print('%s,%s' % (cell.replace(' ', ','), gt_cell_ent[cell])) #输出错误注释的表格，坐标；对应的ground truth
                    print('%s' % (cell.replace(' ', ',')))
        print("false, correct, total")         
        print(len(false_cells), len(correct_cells), len(annotated_cells))            

        precision = len(correct_cells) / len(annotated_cells) if len(annotated_cells) > 0 else 0.0
        recall = len(correct_cells) / len(gt_cell_ent.keys())
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        main_score = f1
        secondary_score = precision
        print("f1, precision, recall")
        print('%.3f %.3f %.3f' % (f1, precision, recall))

        """
    Do something with your submitted file to come up
    with a score and a secondary score.
    if you want to report back an error to the user,
    then you can simply do :
      `raise Exception("YOUR-CUSTOM-ERROR")`
     You are encouraged to add as many validations as possible
     to provide meaningful feedback to your users
    """
        _result_object = {
            "score": main_score,
            "score_secondary": secondary_score
        }
        return _result_object


if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    # answer_file_path = "./DataSets/ToughTablesR2-DBP/Valid/gt/cea_gt.csv"
    # d = './DataSets/ToughTablesR2-DBP/Submissions/cea/'
    answer_file_path = "../../CTACEA_g/ToughTablesR2-DBP/DataSets/ToughTablesR2-DBP/Valid/gt/cea_gt.csv"
    d = '../../CTACEA_g/ToughTablesR2-DBP/test_result/cea'
    for ff in os.listdir(d):
        _client_payload = {}
        # print(ff)
        _client_payload["submission_file_path"] = os.path.join(d, ff)
        _client_payload["aicrowd_submission_id"] = 1123
        _client_payload["aicrowd_participant_id"] = 1234

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = CEA_Evaluator(answer_file_path)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        # print(result)
