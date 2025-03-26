import os
import pandas as pd
from typing import List


class DataLoader:
    def __init__(self, data_name: str):
        self.data_name = data_name
        self.data_path = self.get_data_path()


    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = f.read()

    
    def load_data_queries(self) -> List[str]:
        """
        Load questions from CSV file and filter by corpus_id
        :return: list of questions for the current corpus
        """
        questions_path = os.path.join(os.path.dirname(__file__), "data/questions_df.csv")
        df = pd.read_csv(questions_path)
        filtered_df = df[df['corpus_id'] == self.data_name]
        self.data_queries = filtered_df['question'].tolist()


    def get_data_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), f"data/corpora/{self.data_name}.md")
