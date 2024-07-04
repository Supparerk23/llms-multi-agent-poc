import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { AttributeInfo } from "langchain/schema/query_constructor"
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai"
import { SelfQueryRetriever } from "langchain/retrievers/self_query"
import { FunctionalTranslator } from "langchain/retrievers/self_query/functional"
import { Document } from "@langchain/core/documents"

// import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";

import { createRetrieverTool } from "langchain/tools/retriever";

const docs = [
  new Document({
    pageContent:
      "A bunch of scientists bring back dinosaurs and mayhem breaks loose",
    metadata: { year: 1993, rating: 7.7, genre: "science fiction" },
  }),
  new Document({
    pageContent:
      "Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
    metadata: { year: 2010, director: "Christopher Nolan", rating: 8.2 },
  }),
  new Document({
    pageContent:
      "A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
    metadata: { year: 2006, director: "Satoshi Kon", rating: 8.6 },
  }),
  new Document({
    pageContent:
      "A bunch of normal-sized women are supremely wholesome and some men pine after them",
    metadata: { year: 2019, director: "Greta Gerwig", rating: 8.3 },
  }),
  new Document({
    pageContent: "Toys come alive and have a blast doing so",
    metadata: { year: 1995, genre: "animated" },
  }),
  new Document({
    pageContent: "Three men walk into the Zone, three men walk out of the Zone",
    metadata: {
      year: 1979,
      director: "Andrei Tarkovsky",
      genre: "science fiction",
      rating: 9.9,
    },
  }),
]

const attributeInfo: AttributeInfo[] = [
  {
    name: "genre",
    description: "The genre of the movie",
    type: "string or array of strings",
  },
  {
    name: "year",
    description: "The year the movie was released",
    type: "number",
  },
  {
    name: "director",
    description: "The director of the movie",
    type: "string",
  },
  {
    name: "rating",
    description: "The rating of the movie (1-10)",
    type: "number",
  },
  {
    name: "length",
    description: "The length of the movie in minutes",
    type: "number",
  },
]

const embeddings = new OpenAIEmbeddings()

const documentContents = "Brief summary of a movie"

export const inMemMovieRetriever = async (llm : any) => {

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings)

  const selfQueryRetriever = SelfQueryRetriever.fromLLM({
    llm,
    vectorStore,
    documentContents,
    attributeInfo,
    structuredQueryTranslator: new FunctionalTranslator(),
  })

  // const query1 = await selfQueryRetriever.invoke(
  // "Which movies are less than 90 minutes?"
  // );
  // console.log("query1", query1);

  // const query2 = await selfQueryRetriever.invoke(
  //   "Which movies are rated higher than 8.5?"
  // );

  // console.log("query2", query2);

  // const query3 = await selfQueryRetriever.invoke(
  //   "Which movies are directed by Greta Gerwig?"
  // );
  // console.log("query3", query3);

  // const query4 = await selfQueryRetriever.invoke(
  //   "Which movies are either comedy or drama and are less than 90 minutes?"
  // );
  // console.log("query4", query4);

  // await awaitAllCallbacks();

  const retrieverTool = createRetrieverTool(selfQueryRetriever, {
    name: "movies_search",
    // description:
      // "Search for information about movies. For any questions about movies, you must use this tool!",
    description:
      "useful for what you want to answer queries about movies",
  });

  return retrieverTool

}

// type RagSelfQueryRetrieverReturnType = Awaited<ReturnType<typeof ragSelfQueryRetriever>>

