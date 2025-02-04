# 导入环境变量
from dotenv import load_dotenv
import os
import ast
from dotenv import dotenv_values
from openai import OpenAI
from langchain.schema import HumanMessage
import csv
import json
from collections import Counter
import re
import requests
import pandas as pd
from collections import defaultdict
from langchain.tools import Tool
import xml.etree.ElementTree as ET
from decimal import Decimal
# from langchain.tools.bing_search import BingSearchAPIWrapper
load_dotenv()

# 加载环境变量

config = dotenv_values("key.env")
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
os.environ['OPENAI_API_BASE'] = config["OPENAI_API_BASE"]
# os.environ['QWEN_API_BASE'] = config["QWEN_API_BASE"],
os.environ['SERPAPI_API_KEY'] = config["SERPAPI_API_KEY"]
os.environ['BING_API_KEY'] = config["BING_API_KEY"]


# 验证 API key 是否正确加载
# print(f"Loaded API Key: {os.environ['OPENAI_API_KEY']}")
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
BING_API_KEY = os.environ['BING_API_KEY']


# 初始化大模型
from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-3.5-turbo',
#              temperature=0.5)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model='gpt-3.5-turbo',
             temperature=0.5)


def save_results(label,file_path): #列表，文件路径 将结果按行写入文件
# 使用'w'模式打开文件，表示写入模式
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(label)

    print(f"内容已写入到文件 {file_path}")

def read_file_noheader(file_path): #列表，文件路径 将文件按列读入列表 不读取第一行
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
        reader = csv.reader(file)
        headers = next(reader)  # 读取表头

        # 初始化每列的列表，列数等于表头数量,从表头开始存储每行数据
        # columns = [[header] for header in headers] 
        columns = [[] for _ in headers]

        # 将每个单元格添加到对应列的列表
        for row in reader:
            for i, value in enumerate(row):
                columns[i].append(value)
    return columns

def read_file_header(file_path): #列表，文件路径 将文件按列读入列表 从第一行（包括表头）开始读取
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
        reader = csv.reader(file)
        headers = next(reader)  # 读取表头

        # 初始化每列的列表，列数等于表头数量,从表头开始存储每行数据
        columns = [[header] for header in headers] 
        # columns = [[] for _ in headers]

        # 将每个单元格添加到对应列的列表
        for row in reader:
            for i, value in enumerate(row):
                columns[i].append(value)
    return columns

def read_file_by_row(file_path): #列表，文件路径 将文件按行写入列表 从第一行开始读取
    rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # headers = next(reader)  # 读取表头

        for row in reader:
            rows.append(row)  # row 是一个列表，存储当前行的列数据

    return rows


def get_column_label_with_llm(summary):
    """
    Use ChatOpenAI to predict the semantic label for a column based on its DBpedia class URIs summary.

    :param summary: A list of tuples (class_uri, count) representing the class URI summary.
    :return: The predicted label for the column.
    """
    # 构建提示模板
    react_prompt = """
    You are a highly intelligent data expert. You are given a list of values from a column in a dataset, 
    and your task is to determine the semantic meaning of the column (The answer is in these given values, 
    and the higher these values, the more likely it is that the answer will be). You are asked to only give
     the dbpedia ontology url and do not explain. If the candidates is null, you give the suitable dbpedia ontology url. Follow these steps:

    1. For each value, considering the meaning.
    2. Print the only answer which in the list of candidates.
    Please select the most suitable dbpedia ontology url for the values from candidates, values:{values}, candidates:{candidates}
    """
    # react_prompt = """  
    # You are a highly intelligent data expert. You are given a cell from a table, 
    # and your task is to select the most suitable dbpedia ontology URl for the colnmn according to the cells in the column.
    # You are asked to only give the DBpedia URL. Don't explain. 
    
    # For example: 
    # Question: Please select the most suitable dbpedia resource URl: the entity is 'The  King of Rock 'n' Roll'.
    # Answer: http://dbpedia.org/ontology/Agent

    # Please give the most suitable dbpedia resource URl: the cells in the column are {column}.

    # """

    # 只取前五个最频繁的类 URI
    # top_classes = summary[:5]
    # values_and_classes = ', '.join([f"{class_uri}: Count: {count}" for class_uri, count in top_classes])
    candidates = summary[0]
    values = summary[1]

    # 格式化提示
    formatted_prompt = react_prompt.format(values=values, candidates = candidates)

    # print(formatted_prompt)

    # 使用 ChatOpenAI 与模型交互
    response = llm([HumanMessage(content=formatted_prompt)])

    # 获取响应内容并返回
    result = response.content.strip()
    return result


def calculate_scores(elements): #综合元素出现频率和次序计算得分
    """
    根据元素的位置分配分数，并将列表中相同元素的分数综合计算。

    Args:
        elements (list): 待评分的元素列表。

    Returns:
        dict: 每个元素的综合分数。
    """
    scores = defaultdict(float)  # 用于存储元素及其综合分数，默认值为 0.0
    max_score = 1.0  # 初始分数
    # decrement = 0.1  # 每个位置分数递减量
    # current_score = 0
    score = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for idx, element in enumerate(elements[:10]):
        # 计算当前位置的分数
        # current_score = max_score - idx * decrement
        
        # if current_score <= 0:  # 分数不能为负
        #     break
        scores[element] += score[idx]  # 将分数累加到对应的元素

    return dict(scores)

# DBpedia查询函数 cta
def query_dbpedia_lookup_cta(keyword, max_results=5):
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
        class_uris = []  # List to hold all class URIs for frequency analysis

        for result in root.findall(".//Result"):
            # Get the class URIs for each result
            classes = [cls.find("URI").text for cls in result.findall(".//Class")]
            class_uris.extend(classes)

        # Count the frequency of each class URI
        # class_counts = Counter(class_uris)

        # # Sort by frequency (highest count first)
        # sorted_classes = class_counts.most_common()

        #评分机制
        class_uris_scores = calculate_scores(class_uris)

        # return sorted_classes
        return class_uris_scores
    else:
        print(f"Error: {response.status_code}")
        print("test")
        return []


#spacy 命名实体识别
import spacy
def spacy(text):
    # 加载英语语言模型
    nlp = spacy.load("en_core_web_sm")

    # # 处理文本
    # text = str(column)
    doc = nlp(text)
    tags = []

    for ent in doc.ents:
        tags.append(ent.label_)

    return tags

#bing拼写纠正
#对一段话进行拼写纠正 对一列进行拼写纠正，返回纠正结果列表
import requests
def bing_spell_correction(query):
    """使用 Bing Search API 进行拼写纠正"""
    endpoint = "https://api.bing.microsoft.com/v7.0/spellcheck"
    headers = {
        "Ocp-Apim-Subscription-Key": BING_API_KEY
    }
    params = {
        "mkt": "en-US",
        "mode": "proof",  # 拼写检查模式
        "text": query     # 查询内容
    }
    
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()  # 抛出异常以便捕获错误
    result = response.json()
    
    # 处理拼写建议
    # words = query.split()  # 将输入句子按空格拆分为单词
    suggestions = []

    # 存储所有单词
    for word in query:
        suggestions.append(word)

    if 'flaggedTokens' in result:
        for token in result['flaggedTokens']:
            original_word = token["token"]
            suggested_word = token["suggestions"][0]["suggestion"] if token["suggestions"] else original_word

            # 找到原单词在 suggestions 中的位置并更新建议
            for i,word in enumerate(suggestions):
                if word == original_word:
                    suggestions[i] = suggested_word
    return suggestions

#自定义工具

# # 初始化 Bing Search API 工具
# bing_search = BingSearchAPIWrapper(
#     bing_subscription_key="BING_API_KEY",
#     bing_search_url="https://api.bing.microsoft.com/v7.0/search"
# )

# # 创建工具列表
# tools = [
#     Tool(
#         name="Bin
# gSearch",
#         func=bing_search.run,
#         description="Useful for answering questions by searching the web."
#     )
# ]

#工具0 数据预处理
#从表格中读取，并将处理结果覆盖原cell
# def data_preprocess(column):
# #step1 spacy命名实体识别 挑最多的命名实体作为该列标签
#     input_dir = r"../ToughTablesR2-DBP/test/0L"
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(input_dir, filename)
#             columns = read_file_noheader(file_path)
#             tags = []
#             column_suggestion = []
#             column_suggestion_table = []
#             for column in columns:
#                 tags = spacy(column) #该列标签
#                 # 使用 Counter 统计每个元素的出现次数
#                 counter = Counter(tags)
#                 # 找到出现次数最多的元素
#                 most_common_element, most_common_count = counter.most_common(1)[0]
#                 tag = most_common_element #该列标签

#                 column_suggestion = bing_spell_correction(column)
#                 column_suggestion_table.append(column_suggestion)
#             save_results(column_suggestion_table,file_path)

#             return column_suggestion_table
#         # for value in column_suggestion:




#step2 实体标签进行数据预处理 使用bing进行拼写纠正和缩写补全

#step3 从补全结果中筛选标签和cell相同的的实体，并选出最相似的的实体作为最终结果




#region 工具1，获取列主题
def get_column_topic_with_llm(input_dir):
   
    # region step1 将表格读取为列表
    # input_dir = input_dir[input_dir.index("/home"):].strip().rstrip('"')  # 从 "/home" 开始截取
    input_dir = r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test/0L"

    # # 配置 ChatOpenAI 对象
    # chat = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY,
    #     temperature=0.0,
    #     model='gpt-4o-mini'
    # )


    # input_dir = r"/home/gengyilin/CTA/ToughTablesR2-DBP/test/0L"
    print(f"input_dir:{input_dir}")
    column_topics_all = [] #所有表的列主题
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            
            print(f"\nProcessing file: {filename}")

            # 读取 CSV 文件
            columns = read_file_noheader(file_path)
          
         #存储注释结果
            column_topics = [] #该表的各列列主题
            column_topic_row= [] #cta_result的一行，[文件名，列号，列主题]
            column_topic_rows = [] #cta_result的所有行
            col_num =0
            for column in columns:
                column = bing_spell_correction(column) #拼写纠正
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

                # 使用 ChatOpenAI 与模型交互 
                response = llm.invoke([HumanMessage(content=formatted_prompt)])
                
                # 获取响应内容并返回
                column_topic = response.content.strip()
                # column.append(column_topic)
                # column.append(str(file_name).rstrip('.csv'))
                column_topics.append(column_topic)
                column_topic_row = [filename.rstrip('.csv'), col_num, column_topic]
                column_topic_rows.append(column_topic_row) 
                col_num = col_num + 1
        
    #  step3 将column_topic存储到每列最后一个元素

            file_path_result =  r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cta/cta_result.csv"
            # file_path_result =  r"C:\Users\15333\WPSDrive\1625506091\WPS企业云盘\华中农业大学\我的企业文档\基于大模型的元数据注册\CTACEA_g\ToughTablesR2-DBP\test_result\cta\cta_result.csv"
            # column_topic_rows = list(zip(*column_topic_rows))
            save_results(column_topic_rows,file_path_result) #写入该表的各列列主题
            column_topics_all.append(column_topics)
    

    return column_topics_all

custom_tool_get_column_topic_with_llm = Tool( #定义工具
    name="GetColumnTopiceWithLlm", 
    func=get_column_topic_with_llm, 
    description="give the topic for every column of the table. "
)
#endregion

#region 工具2，使用llm获取cell的dburl


def get_cell_entity_with_llm(candidates):
    input_dir = r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test/0L"
    column_topics_all = read_file_header(r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cta/cta_result.csv")
    db_candidates_all = read_file_by_row(r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cea_candidates.csv")
    print(f"db_candidates长度{len(db_candidates_all)}")
    table_num = 0
    column_topic_num = 0
    db_candidates_num = 0
    cea_rows_all_table = []  # 存储所有表的 CEA 结果

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            print(f"\nProcessing file: {filename}")

            columns = read_file_noheader(file_path)
            df = pd.read_csv(file_path)
            cea_rows = []

            for col_num, column in enumerate(columns):
                column_topic_value = column_topics_all[2][column_topic_num]
                column = bing_spell_correction(column)
                for row_num, value in enumerate(column):
                    react_prompt = """  
                    You are a highly intelligent data expert. You are given a cell from a table, 
                    and your task is to select the most suitable dbpedia resource URl from candidates according
                    to the semantic meaning of the cell, the column topic of the cell and the other cells in the row.
                    You are asked to only give the DBpedia URL. Don't explain. 
                    
                    For example: 
                    Question: Please select the most suitable dbpedia resource URl: the entity is 'The  King of Rock 'n' Roll'.
                    Answer: http://dbpedia.org/resource/Elvis_Presley

                    Please give the most suitable dbpedia resource URl: the cell is {value}, the candidates of the cell are {candidates}, the column topic of the cell is {column_topic_value}, the other cells in the row is {values_in_the_row}.

                    """
                    
                    if not re.match(r'^-?\d+(\.\d+)?$', value): #如果value不为空 并且value不是数字
                        candidates = db_candidates_all[db_candidates_num][3:]
                        values_in_the_row = df.iloc[row_num] #第row_num行的内容
                        formatted_prompt = react_prompt.format(value = value, column_topic_value = column_topic_value, values_in_the_row = values_in_the_row, candidates = candidates)
                        
                        # 使用 ChatOpenAI 与模型交互
                        response = llm.invoke([HumanMessage(content=formatted_prompt)])
                        # 获取响应内容并返回
                        cell_entity = response.content.strip()
                        # print(f"value:{value}, values in the row{values_in_the_row}")
                        # line = f"{db_candidates_num} {candidates} {filename.rstrip('.csv')} {row_num} {col_num}{cell_entity}"
                        line = f"{filename.rstrip('.csv')} {row_num+1} {col_num} {cell_entity}"
                        cea_rows.append(line.split(' '))
                        db_candidates_num += 1  
                column_topic_num = column_topic_num + 1
            cea_rows_all_table.append(cea_rows)
            save_results(cea_rows, "C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cea/cea_result.csv")
    
    return cell_entity

custom_tool_get_cell_entity_with_llm = Tool( # 定义工具
    name="GetCellEntityWithLlm", 
    func=get_cell_entity_with_llm, 
    # description="For a column, the tools give the dbpedia resource URl for the cell acording to the results of tool 'GetColumnTopiceWithLlm'." 
    description="select the most suitable dbpedia resource URl from dbpedia candidates for every cell. "
)

#endregion

#region 工具3，列类型注释
def get_column_type_with_llm(input_dir): #cea_rows_all_table
    """
    Use ChatOpenAI to predict the semantic label for a column based on its DBpedia class URIs summary.

    :param summary: the columns, the db resource urls of cells in the column.
    :return: The predicted db ontology url for the column.
    """
    input_dir = r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test/0L"
    # column_topics_all = read_file_header(r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test_result/cta/cta_result.csv")
     # table_num = 0
    column_topic_num = 0
    # cta_rows_all_table = []
    for filename in os.listdir(input_dir): #i表序号

        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            
            print(f"\nProcessing file: {filename}")

            
            columns = read_file_noheader(file_path) # 按列读取 CSV 文件为列表
            # df = pd.read_csv(file_path) #读取csv文件，便于读取每行内容
            # 存储该表注释结果
            cta_row = []
            cta_rows = []

            col_num = 0
            for column in columns: 
                column = bing_spell_correction(column)
                if re.sub(r'[\d-]', '', ''.join(map(str, column))): # 将column转换为字符串，并判断列中是否只有数字和-，如果只有数字则跳过该列
                    print(f"\nQuerying DBpedia for column: {col_num}")
                    print(f"\ncolumn:{column}")
                    # values_in_the_row = df.iloc[row_num] #第row_num行的内容
                    # 汇总查询结果
                    class_summary_for_column = Counter()    
                    for row_num, value in enumerate(column[:10]): #该列前10个cell生成ontology候选集
                        if re.sub(r'[\d-]', '', ''.join(map(str, value))): #如果value不为空 并且value不是只有数字和-
                            print(f"  Querying DBpedia for value: {value}")
                            class_summary_for_value = query_dbpedia_lookup_cta(value) #cell查询结果
                            # print(f"class_summary_for_value:{class_summary_for_value}")
                            # 汇总查询结果
                            for class_uri in class_summary_for_value:
                                class_summary_for_column[class_uri] += class_summary_for_value[class_uri]
                    # 打印该列的汇总结果
                    print(f"\nClass URI summary for column '{col_num}':")
                    for class_uri, count in class_summary_for_column.most_common():
                        print(f"  Class URI: {class_uri}, Count: {count}")
                    summary = [class_summary_for_column.most_common()[:5], column]
                    column_type = get_column_label_with_llm(summary)
                    print(f"\nPredicted label for column '{col_num}': {column_type}")

                    line = str(filename).rstrip('.csv') + ' ' + str(col_num) + ' ' +  str(column_type) 
                    cta_row = line.split(' ')
                    cta_rows.append(cta_row)
                col_num = col_num + 1
            # column_topic_num = column_topic_num + 1
            


        # cta_rows_all_table.append(cta_rows)  

        print(f"filename,cta_rows_all_table:{filename},{cta_rows}") 
            #将该表的cea结果写入文件
        file_path_result =  r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cta/cta_result.csv"
        # column_topic_rows = list(zip(*column_topic_rows))
        save_results(cta_rows,file_path_result) #写入该表的各列列主题
             
        # table_num = table_num + 1

    # print(f"label_list:{label_list}")
    return cta_rows
    
custom_tool_get_column_type_with_llm = Tool( # 定义工具
    name="GetColumnTypeWithLlm", 
    func=get_column_type_with_llm, 
    description="give the dbpedia ontology url for every column of the table."
)
#endregion

#region 工具4, DBpedia查询函数
def query_dbpedia_lookup(keyword):
    """
    所有表格,对每张表格的每个cell查询dbpedia lookup,将查询到的前k个结果写入C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cea_candidates.csv,该文件中每行: 1N13JCAT,6,0,candidate1,candidate2,...,candidatek
    """

    """
    Query DBpedia Lookup API for entities matching the keyword and return class URI summary.

    :param keyword: The keyword to search for.
    :param max_results: The maximum number of results to retrieve.
    :return: A sorted list of class URIs and their frequencies.
    """

    input_dir = r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test/0L"
    # column_topics_all = read_file_header(r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test_result/cta/cta_result.csv")
 
    

    # table_num = 0
    for filename in os.listdir(input_dir): 

        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            
            # print(f"\nProcessing file: {filename}")

            
            columns = read_file_noheader(file_path) # 按列读取 CSV 文件为列表
            # df = pd.read_csv(file_path) #读取csv文件，便于读取每行内容
            # 存储该表注释结果
            dbpedia_row = []
            dbpedia_rows = []

            col_num = 0
            for column in columns:      
                
                if column:  # 如果列中有有效数据
                    # print(f"\nQuerying DBpedia for column: {col_num}")
                    row_num = 1
                    for value in column:
                        
                        if not re.match(r'^-?\d+(\.\d+)?$', value): #如果value不为空 并且value不是数字 

                            max_results = 5

                            url = "https://lookup.dbpedia.org/api/search/PrefixSearch"
                            headers = {"Accept": "application/xml"}
                            params = {
                                "query": value,
                                "maxResults": max_results
                            }

                            response = requests.get(url, headers=headers, params=params)

                            if response.status_code == 200:
                                root = ET.fromstring(response.content) 
                                resources = [cls.find("URI").text for cls in root] #一个cell的候选集
                                
                                line = str(filename).rstrip('.csv') + ' ' + str(row_num) + ' ' +  str(col_num) 
                                dbpedia_row = line.split(' ') + resources
                                dbpedia_rows.append(dbpedia_row)

                            else:
                                print(f"Error: {response.status_code}")
                                print("test")
                                return []
                            
                        row_num = row_num + 1
                        

                col_num = col_num + 1
                # column_topic_num = column_topic_num + 1
            #将查询结果写入文件
            file_path_result =  r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test_result/cea_candidates.csv"  
            save_results(dbpedia_rows,file_path_result) #写入该表中各cell的候选url
    return resources

custom_tool_query_dbpedia_lookup = Tool( # 定义工具
    name="QueryDbpediaLookup", 
    func=query_dbpedia_lookup, 
    description="For a cell value, the tools will search the dbpedia url in 'https://lookup.dbpedia.org/'."
)

#endregion


# 设置工具
from langchain.agents import load_tools

# tools = load_tools(["serpapi", "llm-math", "Bing Search"], llm=llm)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools = tools + [custom_tool_get_column_topic_with_llm] + [custom_tool_get_cell_entity_with_llm] + [custom_tool_get_column_type_with_llm] + [custom_tool_query_dbpedia_lookup]
 
# 设置提示模板
from langchain.prompts import PromptTemplate
template = ('''
    'Do your best to answer the following questions. If your ability is not sufficient, you may use the following tools.\n\n'
    '{tools}\n\n
    Use the following format:\n\n'
    'Question: the input question you must answer\n'
    'Thought: you should always think about what to do\n'
    'Action: the action to take, should be one of [{tool_names}]\n'
    'Action Input: the input to the action\n'
    'Observation: the result of the action\n'
    '... (this Thought/Action/Action Input/Observation can repeat N times)\n'
    'Thought: I now know the final answer\n'
    'Final Answer: the final answer to the original input question\n\n'
    'Begin!\n\n'
    'Question: {input}\n'
    'Thought:{agent_scratchpad}' 
    '''
)
prompt = PromptTemplate.from_template(template)

# 初始化Agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)

# 构建AgentExecutor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               handle_parsing_errors=True,
                               verbose=True)

# 执行AgentExecutor

input_dir = r"C:/Users/15333/WPSDrive/1625506091/WPS企业云盘/华中农业大学/我的企业文档/基于大模型的元数据注册/CTACEA_g/ToughTablesR2-DBP/test/0L"
# print(input_dir)

# cell = 'The  King of Rock \'n\' Roll'
agent_executor.invoke({
    "input":
                    # f"Please give the topic for every column of the table and give the dbpedia resource URl for the cell of tables use the dbpedia look up. And after that, select the most suitable dbpedia resource URl from dbpedia candidates for every cell."})
                    # f"Please give the topic for every column of the table, give the most suitable dbpedia resource URl for every cell and give the DBpedia ontology URl for every column. file path: {input_dir}."})
                    # f"Please use the search engines to complete the words, which is related to musician,Memphis : 'The Oz'."})
                    # f"Please give the dbpedia resource URl for the cell of tables use the dbpedia look up. file path: {input_dir}."})
                    f"Please give the dbpedia ontology URl for the column of tables.input_dir:{input_dir}"})



