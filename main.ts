import * as dotenv from 'dotenv'
dotenv.config()
import { ChatOpenAI } from "@langchain/openai"

import type { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";

import { createOpenAIFunctionsAgent,AgentExecutor } from "langchain/agents"

import { inMemMovieRetriever } from './src/tools/rag/movie/inMemMovieRetriever'
import { inMemFundRetriever } from './src/tools/rag/fund/inMemFundRetriever'

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

	const movieRetrieverTool = await inMemMovieRetriever(llm)
	const fundRetrieverTool = await inMemFundRetriever(llm)

	const tools = [movieRetrieverTool,fundRetrieverTool];

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
	  input: "what is Finnomena? told me in 5 word",
	});

	console.log("result1 > ",result1);

	const result2 = await agentExecutor.invoke({
	  input: "Which movies are rated higher than 8.5?",
	});

	console.log("result2 > " ,result2);

	const result3 = await agentExecutor.invoke({
	  input: "How many type of funds we have ?",
	});

	console.log("result3 > " ,result3);

	const result4 = await agentExecutor.invoke({
	  input: "How many type of funds we have ? show only couting",
	});

	console.log("result4 > " ,result4);
}

// simpleAI()

execution()