"""pluto.py is just the plain text2sql chain without routers/ intent classification etc."""

from operator import itemgetter

from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from nomos.prompts import QA_PROMPT, TEXT2SQL_PROMPT
from nomos.retrievers import category_retriever, fewshots_retriever, usecases_retriever

GLOBAL_LLM_MODEL = "gpt-3.5-turbo-16k"
GLOBAL_LLM_TEMPERATURE = 0
GLOBAL_LLM = ChatOpenAI(model=GLOBAL_LLM_MODEL, temperature=GLOBAL_LLM_TEMPERATURE)


def create_text2sql_chain(llm, sql_db, prompt=TEXT2SQL_PROMPT, text2sql_parser=None):
    """Defines the text2sql chain"""

    _text2sql_parser = text2sql_parser or StrOutputParser()
    # What happens below
    # Get the question from the input
    # get k from the input, if none, set to 100
    # get examples from the retriever
    # get relevant categories from the retirever
    # get relevant usecases from the retriever
    text2sql_chain = (
        {
            "question": itemgetter("question"),
            "examples": fewshots_retriever,
            "top_k": lambda x: x.get("top_k", 100),
            "categories": category_retriever,
            "usecases": usecases_retriever,
        }
        | create_sql_query_chain(llm, sql_db, prompt=prompt, k=itemgetter("top_k"))
        | _text2sql_parser
        #   | RunnableLambda(lambda x: x.strip('```sql').strip('```'))
    )

    return RunnablePassthrough.assign(sql_query=text2sql_chain)


def create_answer_chain(llm, prompt=QA_PROMPT, answer_parser=None):
    """Creates the Answer Chain. Takes in the SQL Query, SQL Result, and Question to generate and answer"""
    _answer_parser = answer_parser or StrOutputParser()

    answer_chain = prompt | llm | _answer_parser

    return RunnablePassthrough.assign(answer=answer_chain)


def create_execute_chain(sql_db):
    """Runs the SQL Query"""
    return RunnablePassthrough.assign(
        sql_result=itemgetter("sql_query") | QuerySQLDataBaseTool(db=sql_db)
    )


def build_pluto_chain(sql_db, llm=GLOBAL_LLM, **kwargs):
    """Combines Text2Sql chain, Execute Chain and Answer Chain"""
    answer_llm = kwargs.get("answer_llm", llm)
    answer_prompt = kwargs.get("answer_prompt", QA_PROMPT)
    text2sql_prompt = kwargs.get("text2sql_prompt", TEXT2SQL_PROMPT)

    text2sql_chain = create_text2sql_chain(
        llm=llm, sql_db=sql_db, prompt=text2sql_prompt
    )

    answer_chain = create_answer_chain(answer_llm, answer_prompt)
    execute_chain = create_execute_chain(sql_db)

    pluto_chain = text2sql_chain | execute_chain | answer_chain

    return pluto_chain
