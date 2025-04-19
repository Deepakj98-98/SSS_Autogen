from langgraph.graph import StateGraph, END
import ollama
from typing import TypedDict, List
import asyncio
import aiohttp
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

class Mystate(TypedDict):
    retreived_chunks: List[str]
    rephrased_chunks: List[str]
    model: str
    role: str
    previous_role: str
    previous_question: str
    previous_answer:List[str]
    followup:bool
    current_question: str

async def query_ollama(model, prompt):
    url="http://localhost:11434/api/chat"
    payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url,json=payload) as response:
                data=await response.json()
                return data["message"]["content"].strip()



class RoleSelectorAgent(AssistantAgent):
    async def a_run(self, state_data: Mystate)-> Mystate:
        user_input_role=state_data.get("role")

        role_map={
            "dev": "Software Developer",
            "ba":"Business Analyst",
            "Tester":"Software Quality engineer",
            "management":"Organization Leadership"
        }

        selected_role=role_map.get(user_input_role.lower(),"Business Analyst")
        state_data["role"]=selected_role
        print(f"Selected role is : {selected_role}")
        return state_data

class RephraseAgent(AssistantAgent):
    async def a_run(self,state_data: Mystate)->Mystate:
        few_shot_examples = {
        "Business Analyst": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Business Analyst, this highlights how AI enhances data-driven decision-making, enabling quicker and more accurate insights for business strategy."
            }
        ],
        "Software Developer": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Software Developer, AI tools like ML models and data pipelines can be implemented to automate data analysis tasks, reducing manual processing."
            }
        ],
        "Project Manager": [
            {
                "input": "AI tools improve data analysis speed and accuracy.",
                "output": "As a Project Manager, this means teams can deliver results faster and with more precision, improving overall project timelines and outcomes."
            }
        ]
    }
        chunks=state_data["retreived_chunks"]
        role=state_data.get("role","Business Analyst")
        model=state_data.get("model","mistral")

        examples=few_shot_examples.get(role,[])
        example_text=""
        for example in examples:
            example_text+=f"\nExample :\nOriginal: {example['input']}\n rephrased for {role}: {example['output']}\n"
        
        #rephrased=[]
        async def rephrase_chunk(chunk):
            prompt=f"""You are a {role}. Your task is to **rephrase the provided content** while ensuring it remains **strictly relevant** to your role’s priorities, such as business impact, technical implementation, or decision-making.

                    ### **Instructions:**  
                    - The rephrased content **must** be strictly relevant to the role of {role}.  
                    - **Do not** include phrases like "I am a {role}" or "As a {role}" or "Rephrased for the {role}.  
                    - The response should be **concise**, **accurate**, and **summarized** based on the retrieved information.  
                    - Maintain clarity and coherence while ensuring the key insights remain intact.  
                    - Do not introduce any new information or assumptions.  
                    - Do not provide the few shot examples in response

                    ### **Few-Shot Example(s):**  
                    {example_text}
                    ### **Content to Rephrase:**  
                    \"\"\"{chunk}\"\"\"   

                    [Rephrased response goes here]  

                    ---

                """
            return await query_ollama(model,prompt)
            
        
        tasks=[rephrase_chunk(chunk) for chunk in chunks]
        rephrased=await asyncio.gather(*tasks)
    
        state_data["rephrased_chunks"]=rephrased
        return state_data

class FollowupChecker(AssistantAgent): 
    async def a_run(self, state_data: Mystate) -> Mystate:
        previous_question=state_data.get("previous_question","")
        previous_answer=state_data.get("previous_answer",[])
        new_question=state_data.get("current_question","")
        model=state_data.get("model","mistral")
        role=state_data.get("role","NA")
        prev_role=state_data.get("previous_role", "Business Analyst")
        flat_prev_ans = " ".join(previous_answer) if isinstance(previous_answer, list) else previous_answer
        if prev_role!=role or flat_prev_ans.lower() in ("no prior response","sorry, I do not have any information",""):
            state_data["followup"]=False
            return state_data
        prompt = f"""
                You are an AI Assistant that determines if a new question is a **follow-up** based on the **previous conversation context**.

                ### **Previous Context:**
                - **Previous Question:** "{previous_question}"
                - **Previous Answer:** "{previous_answer}"
                - **New Question:** "{new_question}"

                ### **Decision Rules (Strictly Follow These):**
                1. If the **previous question** is empty, `"no prior response"`, `"None"`, or missing → **Respond only with** `"no"`.
                2. If the **new question** is **directly related** to the previous question (e.g., clarifications, further details, asking for elaboration) → **Respond only with** `"yes"`.
                3. If the **new question introduces a completely new topic** or has **no clear dependency on the previous question**, **respond only with** `"no"`.
                4. If the **previous answer is  "sorry, I do not have any information" **, **respond with "no"** 

                ### **Your Response (STRICT RULES APPLY):**
                - **Only answer with either `"yes"` or `"no"` (NO explanations).**
                """
        
        data=await query_ollama(model,prompt)
        reply=data.lower() == "yes"
        state_data["followup"]=reply
        return state_data
    

class FollowupResponse(AssistantAgent):
    async def a_run(self, state_data:Mystate) -> Mystate:
        previous_response = " ".join(state_data.get("previous_answer", [])) or "No prior response."
        current_question = state_data.get("current_question", "")
        previous_question=state_data.get("previous_question")
        model = "gemma3:1b"
        role = state_data.get("role", "Business Analyst")
        print(f"previous_response: {previous_response}")
        print(f"current question: {current_question}")

        prompt = f"""
            You are a {role} AI assistant. Your task is to answer a follow-up question **ONLY using the given previous response**.  

            ### **Instructions (Follow These Strictly):**  
            2. **If the previous response does not contain relevant information, respond with:**  
            - `"Sorry, I do not have information on this."`  
            3. **Do NOT attempt to infer or assume details beyond the previous response.**  
            4. **You must NOT exceed the scope of the given information.**

            ### **Context:**  
            **Previous Question:**  
            \"\"\"{previous_question}\"\"\"  

            **Previous Response:**  
            \"\"\"{previous_response}\"\"\"  

            **Follow-up Question:**  
            \"\"\"{current_question}\"\"\"  

            ### **Your Answer (Strictly Based on the Context Above):**  
            """
        
        data=await query_ollama(model,prompt)
        res=[data]
        state_data["rephrased_chunks"]=res
        return state_data

class ControllerAgent(UserProxyAgent):
    async def a_run(self,state_data:Mystate):
        state=await followup_checker.a_run(state_data)
        if state["followup"]:
            return await followup_responder.a_run(state_data)
        else:
            return await rephraser.a_run(state_data)

role_agent=RoleSelectorAgent(name="role_selector",llm_config=False) 
followup_checker=FollowupChecker(name="followup_checker",llm_config=False)
followup_responder=FollowupResponse(name="followup_responder",llm_config=False)
rephraser=RephraseAgent(name="Rephraser",llm_config=False)
controller_agent= ControllerAgent(name="Controller",code_execution_config=False)

async def run_chatbot(state):
    state=await role_agent.a_run(state)
    state=await controller_agent.a_run(state)
    print("/n Final Response:")
    for index, text in enumerate(state["rephrased_chunks"],1):
            print(f"\n chunk {index}:\n {text}")
    return state

'''
if __name__ == "__main__":
    state = {
        "retreived_chunks": [
            "AI enables companies to analyze big data quickly.",
            "It allows automation of repetitive tasks, improving efficiency."
        ],
        "rephrased_chunks": [],
        "model": "mistral",
        "role": "dev",
        "previous_role": "ba",
        "previous_question": "",
        "previous_answer": [],
        "followup": False,
        "current_question": "How is AI helpful?"
    }

    asyncio.run(run_chatbot(state))
'''