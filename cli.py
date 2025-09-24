# cli.py
import sys
from rag_core import retrieve_candidates, llm_recommend_and_call_tool

def main():
    query = " ".join(sys.argv[1:]) or "friendship and magic"
    cands = retrieve_candidates(query, top_k=5)
    print(llm_recommend_and_call_tool(query, cands))

if __name__ == "__main__":
    main()
