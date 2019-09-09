import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans


class ClusteringUsCompanies(object):
    def __init__(self):
        self.accuracy = 0.99
        self.data = None
        self.df = None
        self.df_original = None
        self.jobs_dict = []
        self.clusters_dict = {}
        self.jobs_by_companies_dict = {}
        self.num_clusters = 0

    def read_csv(self, path, file_name):
        self.data = pd.read_csv(path + file_name)
        self.df = pd.DataFrame(self.data)

    def change_dataframe_attribute(self, attribute, index, new_value):
        self.df[attribute][index] = new_value

    def generate_jobs_dict(self):
        for index, row in self.df.iterrows():
            job_dict = {'job_id': index, 'job_name': row['job_name'], 'company_name_id': row['company_name_id'],
                        'description': row['description'], 'description_short': row['description_short'],
                        'zip_code': row['zip_code'], 'full_time_employees': row['full_time_employees'],
                        'year_founded': row['year_founded'], 'industry': row['company_category']}
            self.jobs_dict.append(job_dict)

    @staticmethod
    def dump_files(content, file_name):
        with open(file_name, 'w') as outfile:
            json.dump(content, outfile)

    def delete_columns(self, column_list):
        for column in column_list:
            self.df.drop(column, axis=1, inplace=True)

    def encode_cat(self, attribute):
        # Create a label (category) encoder object
        le = preprocessing.LabelEncoder()
        # Fit the encoder to the pandas column
        le.fit(self.df[attribute])
        return le

    def apply_encode_to_dataframe(self, encoder, attribute):
        # Apply the fitted encoder to the pandas column
        self.df[attribute] = encoder.transform(self.df[attribute])

    def decode_dataframe(self, encoder, attribute):
        self.df[attribute] = encoder.inverse_transform(self.df[attribute])

    # Compute the algorithm precision with witnesses
    @staticmethod
    def compute_cluster_score(cluster, witnesses):
        clases_dict = {}
        for post in cluster:
            label = post[witnesses][post.index.values[0]]
            if not str(label) in clases_dict.keys():
                clases_dict[str(label)] = []
                clases_dict[str(label)].append(label)
            else:
                clases_dict[str(label)].append(label)
        bigger_class = 0
        for element in clases_dict:
            if len(clases_dict[element]) > bigger_class:
                bigger_class = len(clases_dict[element])

        return (bigger_class * 1.0) / len(cluster)

    def compute_cluster_dict(self, k):
        km = KMeans(n_clusters=k)
        km.fit(self.df)

        # Get cluster assignment labels
        labels = km.labels_

        self.clusters_dict = {}
        for idx, label in enumerate(labels):
            if not str(label) in self.clusters_dict.keys():
                self.clusters_dict[str(label)] = []
                self.clusters_dict[str(label)].append(self.df_original.loc[[idx]])
            else:
                self.clusters_dict[str(label)].append(self.df_original.loc[[idx]])

    def optimize_num_clusters(self, start):
        num_clusters = start - 1
        mean_score = 0

        while mean_score < self.accuracy:
            num_clusters += 1

            print('Cluster Number: ' + str(num_clusters))

            self.compute_cluster_dict(num_clusters)

            total_score = 0
            for cluster in self.clusters_dict:
                score_1 = self.compute_cluster_score(self.clusters_dict[cluster], 'year_founded')
                score_2 = self.compute_cluster_score(self.clusters_dict[cluster], 'zip_code')
                if score_1 < score_2:
                    score = score_1
                else:
                    score = score_2
                print("    Score para Cluster " + str(cluster) + ": " +
                      str(score))
                total_score = total_score + score
            mean_score = (total_score * 1.0) / num_clusters
            print('    Score Promedio: ' + str(mean_score))

        self.num_clusters = num_clusters

    def generate_jobs_by_company(self):
        jobs_by_companies_list = []
        anonymous_jobs_total = []

        for idx, cluster in enumerate(self.clusters_dict):
            jobs_list = []
            anonymous_jobs = []
            company_name = ''
            description = ''
            description_short = ''
            industry = ''
            for job in self.clusters_dict[cluster]:
                if job.company_name_id[job.index.values[0]] != ' ':
                    job_tmp_dict = {'job_name': job.job_name[job.index.values[0]],
                                    'job_id': str(job.job_id[job.index.values[0]])
                                    }
                    jobs_list.append(job_tmp_dict)
                    company_name = job.company_name_id[job.index.values[0]]
                    description = job.description[job.index.values[0]]
                    description_short = job.description_short[job.index.values[0]]
                    industry = job.company_category[job.index.values[0]]
                else:
                    job_an_tmp_dict = {'job_name': job.job_name[job.index.values[0]],
                                       'job_id': str(job.job_id[job.index.values[0]]),
                                       'suggested_company': '',
                                       'description_short': job.description_short[job.index.values[0]]
                                       }
                    anonymous_jobs.append(job_an_tmp_dict)
            company_tmp_dict = {'company_name_id': company_name, 'description': description,
                                'company_id': str(idx), 'jobs': jobs_list,
                                'description_short': description_short, 'industry': industry}
            for anony_job in anonymous_jobs:
                anony_job['suggested_company'] = company_name

            anonymous_jobs_total += anonymous_jobs

            jobs_by_companies_list.append(company_tmp_dict)

        self.jobs_by_companies_dict = {'companies': jobs_by_companies_list, 'anonymous': anonymous_jobs_total}


if __name__ == '__main__':
    clustering_object = ClusteringUsCompanies()
    clustering_object.read_csv('Dataset/', 'final_us_company_dataset.csv')

    clustering_object.change_dataframe_attribute('company_name_id', 1, 'accwea')
    clustering_object.change_dataframe_attribute('company_name_id', 6, 'allnz')
    clustering_object.change_dataframe_attribute('company_name_id', 13, 'aws')
    clustering_object.change_dataframe_attribute('company_name_id', 12, 'amazon')
    clustering_object.change_dataframe_attribute('company_name_id', 17, 'citi')
    clustering_object.change_dataframe_attribute('company_name_id', 22, 'fujtsu')
    clustering_object.change_dataframe_attribute('company_name_id', 27, 'gmin')
    clustering_object.change_dataframe_attribute('company_name_id', 30, 'git')
    clustering_object.change_dataframe_attribute('company_name_id', 40, 'mlife')
    clustering_object.change_dataframe_attribute('company_name_id', 46, 'ubr')
    clustering_object.change_dataframe_attribute('company_name_id', 51, 'yhoo')
    clustering_object.change_dataframe_attribute('company_name_id', 55, 'zurich')

    clustering_object.df_original = clustering_object.df.copy()

    clustering_object.generate_jobs_dict()

    clustering_object.dump_files(clustering_object.jobs_dict, 'jobs_list.json')

    columns_to_delete = ['Unnamed: 0', 'company_type', 'revenue_source', 'business_model',
                         'description', 'description_short', 'job_id']

    clustering_object.delete_columns(columns_to_delete)

    clustering_object.delete_columns(['revenue_source'])

    le_name = clustering_object.encode_cat('company_name_id')
    le_category = clustering_object.encode_cat('company_category')
    le_job = clustering_object.encode_cat('job_name')

    clustering_object.apply_encode_to_dataframe(le_name, 'company_name_id')
    clustering_object.apply_encode_to_dataframe(le_category, 'company_category')
    clustering_object.apply_encode_to_dataframe(le_job, 'job_name')

    clustering_object.optimize_num_clusters(2)
    # clustering_object.compute_cluster_dict(10)

    clustering_object.decode_dataframe(le_name, 'company_name_id')
    clustering_object.decode_dataframe(le_category, 'company_category')
    clustering_object.decode_dataframe(le_job, 'job_name')

    clustering_object.generate_jobs_by_company()

    clustering_object.dump_files(clustering_object.jobs_by_companies_dict, 'jobs_by_company.json')















