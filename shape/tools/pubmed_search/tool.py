import os
import json
from pymed import PubMed
from metapub import PubMedFetcher
from shape.tools.base import BaseTool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Suppress stderr by redirecting it to /dev/null
import sys
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")


class Pubmed_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Pubmed_Search_Tool",
            tool_description="A tool that searches PubMed Central to retrieve relevant article abstracts based on a given list of text queries. Use this ONLY if you cannot use the other more specific ontology tools.",
            tool_version="1.0.0",
            input_types={
                "queries": "list[str] - list of queries terms for searching PubMed.",
                "max_results": "int - Maximum number of results to return (default: 10, max: 50)."
            },
            output_type="list - List of items matching the search query. Each item consists of the title, abstract, keywords, and URL of the article. If no results found, a string message is returned.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(queries=["scoliosis", "injury"])',
                    "description": "Search for PubMed articles mentioning 'scoliosis' OR 'injury'."
                },
                {
                    "command": 'execution = tool.execute(queries=["COVID", "vaccine", "occupational health"])',
                    "description": "Search for PubMed articles mentioning 'COVID' OR 'vaccine' OR 'occupational health'."
                }
            ],
            user_metadata={
                'limitations': "Try to use shorter and more general search queries."
            }
        )
        self.pubmed = PubMed(tool="MyTool", email="my@email.address")
        self.fetch = PubMedFetcher()

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def search_query(self, query_str, max_results=10):
        return self.pubmed.query(query_str, max_results=max_results)

    def execute(self, queries, max_results=10):
        """
        Execute PubMed search. Uses OR logic by default, with automatic fallback to AND and individual searches if no results.
        
        Args:
            queries: List of search terms
            max_results: Maximum number of results to return (default: 10, max: 50)
        """
        try:
            if not queries or len(queries) == 0:
                error_msg = "**⚠️ Error:** No queries provided. Please provide at least one search query term."
                return {"formatted_output": error_msg, "error": "No queries provided", "items": []}
            
            max_results = min(max_results, 50)
            all_items = []
            
            # Strategy 1: Try OR query first (most common, returns broad results)
            query_parts = [f"{q}[Title/Abstract]" for q in queries]
            query_str = f"({' OR '.join(query_parts)}) AND hasabstract[All Fields]"
            print(f"PubMed search (OR): {query_str}")
            results = self._search_and_process(query_str, max_results)
            if results:
                all_items.extend(results)
            
            # Strategy 2: If no results, try AND query (more specific)
            if len(all_items) == 0 and len(queries) > 1:
                query_str = f"({' AND '.join(query_parts)}) AND hasabstract[All Fields]"
                print(f"PubMed search (AND): {query_str}")
                results = self._search_and_process(query_str, max_results)
                if results:
                    all_items.extend(results)
            
            # Strategy 3: If still no results, try individual searches
            if len(all_items) == 0:
                for q in queries:
                    if len(all_items) >= max_results:
                        break
                    query_str = f"{q}[Title/Abstract] AND hasabstract[All Fields]"
                    print(f"PubMed search (individual): {query_str}")
                    results = self._search_and_process(query_str, max_results - len(all_items))
                    if results:
                        all_items.extend(results)
            
            # Remove duplicates based on PMID
            seen_pmids = set()
            unique_items = []
            for item in all_items:
                pmid = item.get('pmid')
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    unique_items.append(item)
                elif not pmid:
                    unique_items.append(item)
            
            if len(unique_items) == 0:
                error_msg = f"**⚠️ No results found for:** {', '.join(queries)}\n\n**Suggestions:**\n- Try different search terms\n- Simplify the query\n- Check spelling"
                return {"formatted_output": error_msg, "error": "No results", "items": [], "queries": queries}
            
            # Format results
            return self._format_results(unique_items[:max_results])
            
        except Exception as e:
            error_msg = f"**⚠️ Error searching PubMed:** {e}\n\n**Queries:** {queries}\n\n**Suggestions:**\n- Check your internet connection\n- Verify PubMed API is accessible\n- Try simpler search terms"
            print(f"Error searching PubMed: {e}")
            import traceback
            traceback.print_exc()
            return {"formatted_output": error_msg, "error": str(e), "items": [], "queries": queries}
    
    def _search_and_process(self, query_str, max_results):
        """Search PubMed and process results."""
        try:
            results = self.search_query(query_str, max_results=max_results)
            if not results:
                return []
            
            items = []
            for article in results:
                try:
                    article_json = json.loads(article.toJSON())
                    pubmed_id = article_json.get('pubmed_id') or (article_json.get('pubmed_id_list', [None])[0] if article_json.get('pubmed_id_list') else None)
                    
                    if not pubmed_id:
                        continue

                    article_detail = self.fetch.article_by_pmid(pubmed_id)
                    if article_detail:
                        items.append({
                            'pmid': pubmed_id,
                            'title': article_detail.title or article_json.get('title', 'N/A'),
                            'abstract': article_detail.abstract or article_json.get('abstract', 'N/A'),
                            'keywords': article_detail.keywords or article_json.get('keywords', []),
                            'url': article_detail.url or f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                            'authors': article_json.get('authors', []),
                            'publication_date': article_json.get('publication_date', 'N/A'),
                            'journal': article_json.get('journal', 'N/A')
                        })
                except Exception as e:
                    print(f"Error processing article: {e}")
                    continue
            
            return items
        except Exception as e:
            print(f"Error in _search_and_process: {e}")
            return []
    
    def _format_results(self, items):
        """Format search results as readable output."""
        formatted_output = f"**📚 Found {len(items)} PubMed article(s):**\n\n"
        for idx, article in enumerate(items[:10], 1):  # Show first 10 results
            formatted_output += f"**{idx}. {article.get('title', 'N/A')}**\n"
            if article.get('pmid'):
                formatted_output += f"   - **PMID:** {article['pmid']}\n"
                formatted_output += f"   - **URL:** https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/\n"
            elif article.get('url'):
                formatted_output += f"   - **URL:** {article['url']}\n"
            if article.get('journal'):
                formatted_output += f"   - **Journal:** {article['journal']}\n"
            if article.get('publication_date'):
                formatted_output += f"   - **Date:** {article['publication_date']}\n"
            if article.get('abstract'):
                abstract = article['abstract'][:300] + "..." if len(article['abstract']) > 300 else article['abstract']
                formatted_output += f"   - **Abstract:** {abstract}\n"
            if article.get('keywords'):
                keywords = article['keywords'] if isinstance(article['keywords'], list) else [article['keywords']]
                formatted_output += f"   - **Keywords:** {', '.join(keywords[:5])}\n"
            formatted_output += "\n"
        if len(items) > 10:
            formatted_output += f"\n*... and {len(items) - 10} more result(s)*\n"
        
        return {
            "formatted_output": formatted_output,
            "items": items,
            "count": len(items)
        }

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/pubmed_search
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage
    tool = Pubmed_Search_Tool()

    # Queries
    queries = ["COVID occupational health"]

    # Execute the tool
    try:
        execution = tool.execute(queries=queries)
        print(execution)
    except ValueError as e: 
        print(f"Execution failed: {e}")

    print("Done!")


