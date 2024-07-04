import * as dotenv from 'dotenv'
dotenv.config()
import { ChatOpenAI } from "@langchain/openai"

import type { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";

import { createOpenAIFunctionsAgent,AgentExecutor } from "langchain/agents"

import { inMemSelfQueryRetriever } from './src/tools/rag/inMemSelfQuerying'

const llm = new ChatOpenAI({
	apiKey: process.env.OPENAI_API_KEY,
	model: "gpt-3.5-turbo",
  	temperature: 0,
})


const simpleAI = async function() {
	const response = await llm.invoke("what is Finnomena? return in 5 word")
	console.log(response)
	console.log(response.content)
}

const execution = async function(){

	const retrieverTool = await inMemSelfQueryRetriever(llm)

	const tools = [retrieverTool];

	const prompt = await pull<ChatPromptTemplate>(
  		"hwchase17/openai-functions-agent"
	);

	const agent = await createOpenAIFunctionsAgent({
	  llm,
	  tools,
	  prompt,
	});

	const agentExecutor = new AgentExecutor({
	  agent,
	  tools,
	});

	const result1 = await agentExecutor.invoke({
	  input: "hi!",
	});

	console.log("result1 > ",result1);

	const result2 = await agentExecutor.invoke({
	  input: "Which movies are rated higher than 8.5?",
	});

	console.log("result2 > " ,result2);
}

// simpleAI()

execution()