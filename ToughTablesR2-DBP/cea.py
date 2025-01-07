import os
import pandas as pd
import requests
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dotenv import dotenv_values
from SPARQLWrapper import SPARQLWrapper, JSON
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 加载环境变量
config = dotenv_values("key.env")
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
os.environ['OPENAI_API_BASE'] = config["OPENAI_API_BASE"]

# 验证 API key 是否正确加载
print(f"Loaded API Key: {os.environ['OPENAI_API_KEY']}")
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


# DBpedia查询函数
def query_dbpedia_lookup(keyword, max_results):
    """
    Query DBpedia Lookup API for entities matching the keyword and return class URI summary.

    :param keyword: The keyword to search for.
    :param max_results: The maximum number of results to retrieve.
    :return: A sorted list of class URIs and their frequencies.
    """
    url = "https://lookup.dbpedia.org/api/search/PrefixSearch"
    headers = {"Accept": "application/xml"}
    params = {
        "query": keyword,
        "maxResults": max_results
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        root = ET.fromstring(response.content) 
        resources = [cls.find("URI").text for cls in root]
        return resources

    else:
        print(f"Error: {response.status_code}")
        print("test")
        return []

#for a column

    #easy CTA 读取一列内容，使用提示获取列主题 返回
def get_column_type_annotation(column):
    react_prompt = """  
    You are a highly intelligent data expert. You are given a column from a table, 
    and your task is to determine the most suitable topic according
    to the semantic meaning of the cell in the column. Don't explain.

    For example: 
    Question: Please Give the dbpedia resource URl of the entity type of the following entity: ["The  King of Rock 'n' Roll", 'Chairman of the Board', 'Piano Mann']
    Answer: Singer

    Please Give the column topic according to the following entity: {column}

    """
    formatted_prompt = react_prompt.format(column=column)
    print(formatted_prompt)

    # 配置 ChatOpenAI 对象
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
        model='gpt-4o-mini'
    )

    # 使用 ChatOpenAI 与模型交互
    response = chat([HumanMessage(content=formatted_prompt)])
    

    # 获取响应内容并返回
    column_topic = response.content.strip()
    return column_topic
    # for a cell
    #数据预处理 结合列主题和单元格内容补全单元格 


    #调用dbpedia查询前50个db_url

    #使用大模型选择最合适的db_url

    #CTA判断 输出db_url

#写入文件    





# 构建 ChatOpenAI 进行推理
def get_column_label_with_llm(value,column_topic,values_in_the_row):
    """
    Use ChatOpenAI to predict the semantic label for a cell based on its DBpedia class URIs summary.

    :param summary: An entity.
    :return: The predicted label for the cell.
    """

    react_prompt = """  
    You are a highly intelligent data expert. You are given a cell from a table, 
    and your task is to select the most suitable dbpedia resource URl from candidates according
    to the semantic meaning of the cell, the column topic of the cell and the other cells in the row.
    You are asked to give the DBpedia URL. Don't explain.
    
    For example: 
    Question: Please select the most suitable dbpedia resource URl: the entity is Abbeville; the candidates are ['http://dbpedia.org/resource/Abbeville', 'http://dbpedia.org/resource/SC_Abbeville', 'http://dbpedia.org/resource/Abbeville,_South_Carolina', 'http://dbpedia.org/resource/Abbeville,_Louisiana', 'http://dbpedia.org/resource/Abbeville,_Alabama', 'http://dbpedia.org/resource/Abbeville_County,_South_Carolina', 'http://dbpedia.org/resource/Abbeville,_Georgia', 'http://dbpedia.org/resource/Aerodrome_Abbeville', 'http://dbpedia.org/resource/Arrondissement_of_Abbeville', 'http://dbpedia.org/resource/Vermilion_Parish,_Louisiana']; The column topic is City; the other cells in the row is [Abbeville, Georgia, GA].
    Answer: http://dbpedia.org/resource/Elvis_Presley

    Please select the most suitable dbpedia resource URl: the entity is {value}; The column topic is {column_topic}; the other cells in the row is {values_in_the_row}.

    """


    # # 只取前五个最频繁的类 URI
    # top_classes = summary[:5]
    # values_and_classes = ', '.join([f"{class_uri}: Count: {count}" for class_uri, count in top_classes])

    # 格式化提示
    formatted_prompt = react_prompt.format(value = value, column_topic = column_topic,values_in_the_row = values_in_the_row)
    # print("-----------------------------------")
    # print(formatted_prompt)

    # 配置 ChatOpenAI 对象
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
        model='gpt-4o-mini'
    )

    # 使用 ChatOpenAI 与模型交互
    response = chat([HumanMessage(content=formatted_prompt)])
    

    # 获取响应内容并返回
    result = response.content.strip()
    return result


# 遍历目录并处理CSV文件

def process_csv_files(input_dir):
    """
    Process all CSV files in the specified directory, query DBpedia for all values in each column,
    and print the class URI summary for each column, followed by a prediction of its semantic label.

    :param input_dir: Directory containing the CSV files.
    """


    #存储注释结果
    label_list = []
    label_list_row = []


    # 遍历输入目录下的所有CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            print(f"\nProcessing file: {filename}")

            # 读取CSV文件
            df = pd.read_csv(file_path)
            col_num = 0
            # 遍历每列
            for col in df.columns:
                # 过滤掉空值
                non_null_values = df[col].dropna().astype(str).tolist()
                #记录该列去重后的value,以及对应的db_url，如果遇到注释过的value，直接使用之前的结果。避免重复注释，浪费资源。key: cell value , value: db_url
                # valueDict = {}
                
                if non_null_values:  # 如果列中有有效数据
                    print(f"\n------------------------Querying DBpedia for column: {col}-----------------------------")
                    
                    print(f"the cells of the column:{non_null_values}")
                    # # 汇总查询结果
                    # class_summary_for_column = Counter()
                    column_topic = get_column_type_annotation(non_null_values)
                    print(f"the column topic is:{column_topic}")
                    print("-----------------------------------")
                    row_num = 1
                    #遍历该列每个cell
                    for value in non_null_values:

                        
                        #查询结果
                        pattern = r'^-?\d+(\.\d+)?$'
                        if value and (not(re.match(pattern, value))): #如果value不为空 并且value不是数字
                            #判断之前是否预测过该value
                            # if value in valueDict:
                            #     # predicted_label = valueDict[value]
                            #     print(f"该value已预测过")   
                            
                            # else: #之前没有预测过，调用大模型预测该value的db_url
                            # print("调用大模型预测标签")
                            values_in_the_row = df.iloc[row_num-1] #第row_num行的内容
                            # db_resources = query_dbpedia_lookup(value, 20) #查询dbpedia中前20个resource
                            predicted_label = get_column_label_with_llm(value, column_topic, values_in_the_row)
                            # valueDict[value] = str(predicted_label)  #写入该列cell value字典

                            line = str(filename).rstrip('.csv') + ' ' + str(row_num) + ' ' + str(col_num) + ' ' +  str(predicted_label) 
                            label_list_row = line.split(' ')
                            label_list.append(label_list_row)
                            
                            # print(f"Predicted label for cell:\n value:'{value}'\ncandidates: '{db_resources}\n")
                            print(f"column topic: '{column_topic}'\n\nvalues in the row:\n'{values_in_the_row}'\n")
                            print(f"predicted label: {predicted_label}")    
                            # print(f"label_list_row: {label_list_row}\n")
                            print("-----------------------------------")
                        
                        row_num = row_num + 1
                    # print(f"该列value字典：{valueDict}")
                    col_num = col_num + 1 
    return label_list


input_dir_path = r"/home/gengyilin/CTA/ToughTablesR2-DBP/test/1B"

# 执行处理
label = process_csv_files(input_dir_path)


# print(f"label: {label}")
# print(label[0])

#将结果写入文件
file_path = "/home/gengyilin/CTA/ToughTablesR2-DBP/test_result/test.csv"

# 使用'w'模式打开文件，表示写入模式
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(label)

print(f"内容已写入到文件 {file_path}")
