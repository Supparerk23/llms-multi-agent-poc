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
    metadata: { shortCode: "SCBRF", provider: "SCB Asset Management Co., Ltd." , type: "Equity Funds" },
  }),
  new Document({
    pageContent:
      "SCB Short Term Fixed Income",
    metadata: { shortCode: "SCBSFF", provider: "SCB Asset Management Co., Ltd." , type: "Fixed Income Funds" },
  }),
  new Document({
    pageContent:
      "K Fixed Income RMF",
    metadata: { shortCode: "KFIRMF", provider: "Kasikorn Asset Management Co. Ltd" , type: "Fixed Income Funds" },
  }),
  new Document({
    pageContent:
      "K SET 50 Index",
    metadata: { shortCode: "K-SET50", provider: "Krungthai Asset Management PLC" , type: "Mixed Funds" },
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
    description: "the provider company of fund",
    type: "string",
  },
  {
    name: "type",
    description: "type of fund",
    type: "string",
  },
]

const embeddings = new OpenAIEmbeddings()

const documentContents = "Brief summary of a fund"

export const inMemFundRetriever = async (llm : any) => {

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings)

  const selfQueryRetriever = SelfQueryRetriever.fromLLM({
    llm,
    vectorStore,
    documentContents,
    attributeInfo,
    structuredQueryTranslator: new FunctionalTranslator(),
  })

  const retrieverTool = createRetrieverTool(selfQueryRetriever, {
    name: "fund_search",
    description:
      "useful for what you want to answer queries about fund",
  });

  return retrieverTool

}

