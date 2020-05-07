import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import scipy.stats as stats

class Big5():
    def __init__(self):
        self.df = pd.read_csv('data/data-finalV2.csv')
        self.df = self.df.head(6000)
        self.prep_df()
        self.questions_key = {
			'I am the life of the party.': 'E1',
			"I don't talk a lot.": 'E2',
			'I feel comfortable around people.': 'E3',
			'I keep in the background.': 'E4',
			'I start conversations.': 'E5',
			'I have little to say.': 'E6',
			'I talk to a lot of different people at parties.': 'E7',
			"I don't like to draw attention to myself.": 'E8',
			"I don't mind being the center of attention.": 'E9',
			'I am quiet around strangers.': 'E10',
			'I get stressed out easily.': 'N1',
			'I am relaxed most of the time.': 'N2',
			'I worry about things.': 'N3',
			'I seldom feel blue.': 'N4',
			'I am easily disturbed.': 'N5',
			'I get upset easily.': 'N6',
			'I change my mood a lot.': 'N7',
			'I have frequent mood swings.': 'N8',
			'I get irritated easily.': 'N9',
			'I often feel blue.': 'N10',
			'I feel little concern for others.': 'A1',
			'I am interested in people.': 'A2',
			'I insult people.': 'A3',
			"I sympathize with others' feelings.": 'A4',
			"I am not interested in other people's problems.": 'A5',
			'I have a soft heart.': 'A6',
			'I am not really interested in others.': 'A7',
			'I take time out for others.': 'A8',
			"I feel others' emotions.": 'A9',
			'I make people feel at ease.': 'A10',
			'I am always prepared.': 'C1',
			'I leave my belongings around.': 'C2',
			'I pay attention to details.': 'C3',
			'I make a mess of things.': 'C4',
			'I get chores done right away.': 'C5',
			'I often forget to put things back in their proper place.': 'C6',
			'I like order.': 'C7',
			'I shirk my duties.': 'C8',
			'I follow a schedule.': 'C9',
			'I am exacting in my work.': 'C10',
			'I have a rich vocabulary.': 'O1',
			'I have difficulty understanding abstract ideas.': 'O2',
			'I have a vivid imagination.': 'O3',
			'I am not interested in abstract ideas.': 'O4',
			'I have excellent ideas.': 'O5',
			'I do not have a good imagination.': 'O6',
			'I am quick to understand things.': 'O7',
			'I use difficult words.': 'O8',
			'I spend time reflecting on things.': 'O9',
			'I am full of ideas.': 'O10',
		}

    def handle_personality_test(self, answers):
        answer_dict = {}
        for question, answer in answers.items():
            key = self.questions_key[question]
            answer_dict[key] = answer
            
        score_dict = {'O_score': 0, 'C_score': 0, 'E_score': 0, 'A_score': 0, 'N_score': 0}
        for trait_key, answer in answer_dict.items():
            if 'O' in trait_key:
                score_dict['O_score'] += answer
            if 'C' in trait_key:
                score_dict['C_score'] += answer
            if 'E' in trait_key:
                score_dict['E_score'] += answer
            if 'A' in trait_key:
                score_dict['A_score'] += answer
            if 'N' in trait_key:
                score_dict['N_score'] += answer	

        for key, score in score_dict.items():
            score_dict[key] = score/10
        
        perc_dict = {}
        for key, score in score_dict.items():
            if key == 'O_score':
                perc = stats.percentileofscore(self.df[key], score)
                perc_dict['O_perc'] = perc
            if key == 'C_score':
                perc = stats.percentileofscore(self.df[key], score)
                perc_dict['C_perc'] = perc
            if key == 'E_score':
                perc = stats.percentileofscore(self.df[key], score)
                perc_dict['E_perc'] = perc
            if key == 'A_score':
                perc = stats.percentileofscore(self.df[key], score)
                perc_dict['A_perc'] = perc
            if key == 'N_score':
                perc = stats.percentileofscore(self.df[key], score)
                perc_dict['N_perc'] = perc
        
        result_dict = {}
        result_dict['percentiles'] = perc_dict
        result_dict['scores'] = score_dict
        
        rec_list = self.get_list_recommendations(score_dict)
        print(rec_list)

        result_dict["recommendations"] = rec_list
        
        return(result_dict)

    def get_list_recommendations(self, sim_scores):
        user_score = np.array([sim_scores['O_score'], sim_scores['C_score'], sim_scores['E_score'], sim_scores['A_score'], sim_scores['N_score']])
        
        all_user_sim = pd.read_csv('data/personality_predictions.csv')

        arr1 = all_user_sim.values

        cosine_sim = []

        for row in arr1:
            comp_row = row[1:]
            sim = np.dot(user_score, comp_row)/ (np.linalg.norm(user_score) * np.linalg.norm(comp_row))
            
            cosine_sim.append(sim)
        np_cosine = np.array(cosine_sim).reshape(-1,1)


        sims = np.hstack((arr1,np_cosine))
        sims_df = pd.DataFrame(data=sims,columns=['UserIds', 'sOPN', 'sCON', 'sEXT', 'sAGR', 'sNEU', 'similarity_scores'])
        sorted_sims_df = sims_df.sort_values(by=["similarity_scores"], ascending=False)
        top_rec = sorted_sims_df.iloc[:10]

        final_list = []
        top_np = top_rec.to_numpy()

        for rec in top_np:
            rec_user = {}
            rec_user['UserId'] = rec[0]
            rec_user['sOPN'] = rec[1]
            rec_user['sCON'] = rec[2]
            rec_user['sEXT'] = rec[3]
            rec_user['sAGR'] = rec[4]
            rec_user['sNEU'] = rec[5]
            final_list.append(rec_user)
        
        return final_list
    
    def calc_score(self, df):
        score = []
        for row in df.values:
            score.append(row.mean())
        return score
        
    def prep_df(self):
        O_columns = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
        C_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
        E_columns = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
        A_columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
        N_columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10']
        
        self.df['O_score'] = self.calc_score(self.df[O_columns])
        self.df['C_score'] = self.calc_score(self.df[C_columns])
        self.df['E_score'] = self.calc_score(self.df[E_columns])
        self.df['A_score'] = self.calc_score(self.df[A_columns])
        self.df['N_score'] = self.calc_score(self.df[N_columns])