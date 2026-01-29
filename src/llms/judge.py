from typing import List, Dict

class GenericJudgeLLM:
    def __init__(self, any_llm):
        self.llm = any_llm

    def generate(self, prompt: str):
        raw = self.llm.generate(prompt)
        return raw

    async def a_generate(self, prompt: str):
        return await self.llm.a_generate(prompt)


class ChatJudgeLLM:
    def __init__(self, any_llm, system_prompt: str = None):
        self.llm = any_llm
        self.conversation_history: List[Dict[str, str]] = []
        
        # Set up system prompt if provided
        if system_prompt:
            self.conversation_history.append({
                "role": "system", 
                "content": system_prompt
            })
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response using chat completion
        response = self.llm.generate_chat_completion(
            messages=self.conversation_history,
            **kwargs
        )
        
        # Add assistant response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

    async def a_generate(self, prompt: str, **kwargs) -> str:
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response using async chat completion
        response = await self.llm.a_generate_chat_completion(
            messages=self.conversation_history,
            **kwargs
        )
        
        # Add assistant response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        system_prompt = None
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_prompt = self.conversation_history[0]["content"]
        
        self.conversation_history = []
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_system_message(self, message: str):
        # Remove existing system message if any
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history.pop(0)
        
        # Add new system message at the beginning
        self.conversation_history.insert(0, {
            "role": "system",
            "content": message
        })
    
    def get_conversation_summary(self) -> str:
        summary = f"Conversation has {len(self.conversation_history)} messages:\n"
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"{i+1}. {role}: {content}\n"
        return summary
