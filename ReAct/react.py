# 导入环境变量
from dotenv import load_dotenv
import os
import ast
from dotenv import dotenv_values
from openai import OpenAI
from langchain.schema import HumanMessage
import csv
import json
import re
load_dotenv()

# 加载环境变量

config = dotenv_values("key.env")
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
os.environ['OPENAI_API_BASE'] = config["OPENAI_API_BASE"]
# os.environ['QWEN_API_BASE'] = config["QWEN_API_BASE"],
os.environ['SERPAPI_API_KEY'] = config["SERPAPI_API_KEY"]

# 验证 API key 是否正确加载
# print(f"Loaded API Key: {os.environ['OPENAI_API_KEY']}")
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 初始化大模型
from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-3.5-turbo',
#              temperature=0.5)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model='gpt-3.5-turbo',
             temperature=0.5)

def save_results(label,file_path): #列表，文件路径 将结果按行写入文件
# 使用'w'模式打开文件，表示写入模式
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(label)

    print(f"内容已写入到文件 {file_path}")

def read_file(file_path): #列表，文件路径 将文件按列写入列表 不读取第一行
    with open(file_path, 'r', encoding='utf-8') as file:
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

def read_file_header(file_path): #列表，文件路径 将文件按列写入列表 从第一行（包括表头）开始读取
    with open(file_path, 'r', encoding='utf-8') as file:
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

#自定义工具
#定义函数
def my_custom_tool(input_text: str) -> str:
    # 这里可以是任何自定义逻辑，例如查询数据库或调用外部 API
    return f"自定义工具调用成功，你输入的是：{input_text}"

# 定义工具
from langchain.tools import Tool
custom_tool = Tool(
    name="MyCustomTool", 
    func=my_custom_tool, 
    description="A tool that returns the input text. Use this for demonstration."
)

#工具1，将表格按列切片为列表

#工具2，获取列主题
def get_column_topic_with_llm(input_dir):
   
    # region step1 将表格读取为列表
    input_dir = input_dir[input_dir.index("/home"):].strip().rstrip('"')  # 从 "/home" 开始截取

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
            columns = read_file(file_path)
          
         #存储注释结果
            column_topics = [] #该表的各列列主题
            column_topic_row= [] #cta_result的一行，[文件名，列号，列主题]
            column_topic_rows = [] #cta_result的所有行
            col_num =0
            for column in columns:
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
        
    #region step3 将column_topic存储到每列最后一个元素

            file_path_result =  r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test_result/cta/cta_result.csv"
            # column_topic_rows = list(zip(*column_topic_rows))
            save_results(column_topic_rows,file_path_result) #写入该表的各列列主题
            column_topics_all.append(column_topics)
    #endregion

    return column_topics_all


# 定义工具
custom_tool_get_column_topic_with_llm = Tool(
    name="GetColumnTopiceWithLlm", 
    func=get_column_topic_with_llm, 
    description="For a column, the tool will give the column topic of each column. It will return a label, in which the first value is filename, the second value is the column topic, and the third value is the cells of the column. "
)


# 工具3，使用llm获取cell的dburl
def get_cell_entity_with_llm(column_topics_all):
    """
    Use ChatOpenAI to predict the semantic label for a cell based on its DBpedia class URIs summary.

    :param summary: An entity.
    :return: The predicted label for the cell.
    """
    input_dir = r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test/0L"
    column_topics_all = read_file_header(r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test_result/cta/cta_result.csv")
 
    

    table_num = 0
    column_topic_num = 0
    for filename in os.listdir(input_dir): #i表序号
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            
            print(f"\nProcessing file: {filename}")

            # 读取 CSV 文件
            columns = read_file(file_path)
            
            #存储该表注释结果
            cea_row = []
            cea_rows = []

            col_num = 0
            for column in columns: # j列序号
                # column_topic = column_topics[col_topic_num][2]
                # column_topic_value = column_topics_all[table_num][col_num]
                column_topic_value = column_topics_all[2][column_topic_num]
                row_num = 0
                for value in column: #遍历所有cell
                                      
                    react_prompt = """  
                    You are a highly intelligent data expert. You are given a cell from a table, 
                    and your task is to select the most suitable dbpedia resource URl from candidates according
                    to the semantic meaning of the cell, the column topic of the cell and the other cells in the row.
                    You are asked to only give the DBpedia URL. Don't explain. 
                    
                    For example: 
                    Question: Please select the most suitable dbpedia resource URl: the entity is 'The  King of Rock 'n' Roll'.
                    Answer: http://dbpedia.org/resource/Elvis_Presley

                    Please give the most suitable dbpedia resource URl: the entity is {value}, the cell of column topic is {column_topic_value}.

                    """
                    pattern = r'^-?\d+(\.\d+)?$'
                    if value and (not(re.match(pattern, value))): #如果value不为空 并且value不是数字 
                        formatted_prompt = react_prompt.format(value = value, column_topic_value = column_topic_value)
                        # print("-----------------------------------")
                        # print(formatted_prompt)

                        # 使用 ChatOpenAI 与模型交互
                        response = llm.invoke([HumanMessage(content=formatted_prompt)])
                        row_num = row_num + 1
                        # 获取响应内容并返回
                        cell_entity = response.content.strip()
                        line = str(filename).rstrip('.csv') + ' ' + str(row_num) + ' ' + str(col_num) + ' ' +  str(cell_entity) 
                        cea_row = line.split(' ')
                        cea_rows.append(cea_row)
                    
                col_num = col_num + 1 
                column_topic_num = column_topic_num + 1
            
            #将该表的cea结果写入文件
            file_path_result =  r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test_result/cea/cea_result.csv"
            # column_topic_rows = list(zip(*column_topic_rows))
            save_results(cea_rows,file_path_result) #写入该表的各列列主题
        table_num = table_num + 1

    # print(f"label_list:{label_list}")
    return cea_rows
    
# 定义工具
custom_tool_get_cell_entity_with_llm = Tool(
    name="GetCellEntityWithLlm", 
    func=get_cell_entity_with_llm, 
    description="For a column, the tools give the dbpedia resource URl for the cell acording to the results of tool 'GetColumnTopiceWithLlm'."
)

#工具4，列类型注释


#工具5，将cea结果写入文件

#工具6，将cta结果写入文件


# 设置工具
from langchain.agents import load_tools
# from langchain_community.agent_toolkits.load_tools import load_tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools = tools + [custom_tool] + [custom_tool_get_cell_entity_with_llm] + [custom_tool_get_column_topic_with_llm]
 
# 设置提示模板
from langchain.prompts import PromptTemplate
template = ('''
    '尽你所能用中文回答以下问题。如果能力不够你可以使用以下工具:\n\n'
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

input_dir = r"/home/gengyilin/CTACEA/ToughTablesR2-DBP/test/0L"

print(input_dir)
agent_executor.invoke({
    "input":
                    #    f"Please give the most suitable dbpedia resource URl for every cell, give the most suitable dbpedia ontology url for every column of the table and save the final answer in the file. The file path of the table is {file_path}."})
                    f"Please give the topic for every column of the table and give the most suitable dbpedia resource URl for every cell. file path: {input_dir}."})
                    # f"Please give the topic for every column of the table, file path: {input_dir}."})
