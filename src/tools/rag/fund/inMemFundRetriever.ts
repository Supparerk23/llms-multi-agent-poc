import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { AttributeInfo } from "langchain/schema/query_constructor"
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai"
import { SelfQueryRetriever } from "langchain/retrievers/self_query"
import { FunctionalTranslator } from "langchain/retrievers/self_query/functional"
import { Document } from "@langchain/core/documents"

import { createRetrieverTool } from "langchain/tools/retriever";

const docs = [
  new Document({
    pageContent:
      "SCB Retirement Fixed Income",
    metadata: { shortCode: "SCBRF", provider: "SCB" ,company: "SCB Asset Management Co., Ltd." , type: "Equity Funds" },
  }),
  new Document({
    pageContent:
      "SCB Short Term Fixed Income",
    metadata: { shortCode: "SCBSFF", provider: "SCB" ,company: "SCB Asset Management Co., Ltd." , type: "Fixed Income Funds" },
  }),
  new Document({
    pageContent:
      "K Fixed Income RMF",
    metadata: { shortCode: "KFIRMF", provider: "Kasikorn" ,company: "Kasikorn Asset Management Co. Ltd" , type: "Fixed Income Funds" },
  }),
  new Document({
    pageContent:
      "K SET 50 Index",
    metadata: { shortCode: "K-SET50", provider: "Krungthai" ,company: "Krungthai Asset Management PLC" , type: "Mixed Funds" },
  }),
]

const attributeInfo: AttributeInfo[] = [
  {
    name: "shortCode",
    description: "the code of fund",
    type: "string",
  },
  {
    name: "provider",
    description: "the provider of fund",
    type: "string",
  },
  {
    name: "company",
    description: "the full name of provider company of fund",
    type: "string",
  },
  {
    name: "type",
    description: "type of fund",
    type: "string",
  },
]

const embeddings = new OpenAIEmbeddings()
const documentContents = "List of a funds"
// const documentContents = "Brief summary of a funds"
// const documentContents = "Brief summary of a fund"

export const inMemFundRetriever = async (llm : any) => {

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings)

  const selfQueryRetriever = SelfQueryRetriever.fromLLM({
    llm,
    vectorStore,
    documentContents,
    attributeInfo,
    structuredQueryTranslator: new FunctionalTranslator(),
  })

  // const query1 = await selfQueryRetriever.invoke(
  //   "How many funds from provider SCB we have ?"
  // );

  // console.log("query1", query1);

  const retrieverTool = createRetrieverTool(selfQueryRetriever, {
    name: "funds_search",
    // description:
    //   "useful for what you want to answer queries about funds",
    description:
      "Search for information about funds. For any questions about funds, you must use this tool!",
  });

  return retrieverTool

}

