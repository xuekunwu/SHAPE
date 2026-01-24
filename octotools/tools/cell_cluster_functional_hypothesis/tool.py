import os
import json
from typing import List, Dict, Any, Optional
from octotools.tools.base import BaseTool
from octotools.engine.openai import ChatOpenAI
from octotools.tools.pubmed_search.tool import Pubmed_Search_Tool


class Cell_Cluster_Functional_Hypothesis_Tool(BaseTool):
    require_llm_engine = True
    require_api_key = True

    def __init__(self, model_string="gpt-4o-mini", api_key=None):
        super().__init__(
            tool_name="Cell_Cluster_Functional_Hypothesis_Tool",
            tool_description="A specialized tool for generating functional state hypotheses for cell clusters based on their top regulator genes. Follows strict evidence hierarchy and uncertainty principles. Outputs structured annotations with confidence scores and evidence citations.",
            tool_version="1.0.0",
            input_types={
                "clusters": "list[dict] - List of cluster dictionaries, each containing: cluster_id (str), regulator_genes (list[str]), tissue_context (str, optional), disease_context (str, optional), target_lineage (str, optional)",
                "max_literature_per_gene": "int - Maximum number of literature results per gene (default: 5)",
                "include_grouping": "bool - Whether to perform functional class grouping (default: True)"
            },
            output_type="dict - Structured output containing functional state hypotheses, confidence scores, evidence, and formatted text for each cluster",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(clusters=[{"cluster_id": "Cluster_1", "regulator_genes": ["NKX2-1", "SFTPC"], "tissue_context": "lung", "target_lineage": "epithelial"}])',
                    "description": "Generate functional hypothesis for a lung epithelial cluster with NKX2-1 and SFTPC markers."
                }
            ],
            user_metadata={
                "limitation": "This tool generates hypotheses based on available evidence, not definitive annotations. Confidence scores reflect plausibility, not correctness.",
                "best_practice": "1) Provide complete cluster information including tissue/disease context when available. 2) Ensure regulator_genes are correctly spelled. 3) Review evidence citations for validation. 4) Use confidence scores to assess hypothesis reliability."
            }
        )
        self.model_string = model_string
        self.api_key = api_key
        self.pubmed_tool = Pubmed_Search_Tool()
        
        # Evidence hierarchy weights
        self.EVIDENCE_WEIGHTS = {
            "hard_lineage_marker": 0.40,
            "functional_program_coherence": 0.30,
            "tissue_disease_context": 0.20,
            "conflicting_signals": -0.10
        }

    def _collect_evidence(self, genes: List[str], tissue_context: Optional[str] = None, 
                         disease_context: Optional[str] = None, max_results: int = 5,
                         use_llm_knowledge: bool = True) -> Dict[str, Any]:
        """
        Collect evidence using efficient hybrid strategy:
        1. Use LLM knowledge for common genes (fast, no API calls)
        2. Use batch PubMed queries for comprehensive evidence (fewer API calls)
        
        Args:
            genes: List of gene names
            tissue_context: Optional tissue context
            disease_context: Optional disease context
            max_results: Max results per query type
            use_llm_knowledge: Whether to use LLM knowledge first (default: True)
        """
        all_evidence = {
            "hard_lineage_markers": [],
            "functional_programs": [],
            "tissue_disease_context": [],
            "conflicting_signals": []
        }
        
        if not genes:
            return all_evidence
        
        # Strategy 1: Use LLM knowledge for quick initial evidence (if enabled)
        if use_llm_knowledge and len(genes) <= 10:  # Only for reasonable number of genes
            llm_evidence = self._get_llm_gene_knowledge(genes, tissue_context, disease_context)
            # Merge LLM evidence
            for evidence_type in ["hard_lineage_markers", "functional_programs", "tissue_disease_context"]:
                all_evidence[evidence_type].extend(llm_evidence.get(evidence_type, []))
        
        # Strategy 2: Batch PubMed queries for comprehensive evidence
        # Only query PubMed if we need more evidence or LLM knowledge is insufficient
        need_pubmed = not use_llm_knowledge or len(all_evidence["hard_lineage_markers"]) < len(genes)
        
        if need_pubmed:
            # Batch query for hard lineage markers (all genes + "lineage marker")
            print(f"Batch querying PubMed for lineage markers ({len(genes)} genes)...")
            lineage_queries = genes + ["lineage marker", "cell type marker"]
            result = self.pubmed_tool.execute(queries=lineage_queries, max_results=max_results * len(genes))
            if result.get("items"):
                for item in result["items"]:
                    # Match which gene(s) this article is about
                    matched_genes = [g for g in genes if g.lower() in (item.get("title", "") + " " + item.get("abstract", "")).lower()]
                    for gene in matched_genes:
                        all_evidence["hard_lineage_markers"].append({
                            "gene": gene,
                            "source": "PubMed",
                            "pmid": item.get("pmid"),
                            "url": item.get("url"),
                            "title": item.get("title"),
                            "abstract": item.get("abstract", "")[:200] + "..." if item.get("abstract") else "",
                            "evidence_type": "hard_lineage_marker"
                        })
            
            # Batch query for functional programs (all genes + "function")
            print(f"Batch querying PubMed for functional programs ({len(genes)} genes)...")
            function_queries = genes + ["function", "biological process"]
            result = self.pubmed_tool.execute(queries=function_queries, max_results=max_results * len(genes))
            if result.get("items"):
                for item in result["items"]:
                    matched_genes = [g for g in genes if g.lower() in (item.get("title", "") + " " + item.get("abstract", "")).lower()]
                    for gene in matched_genes:
                        all_evidence["functional_programs"].append({
                            "gene": gene,
                            "source": "PubMed",
                            "pmid": item.get("pmid"),
                            "url": item.get("url"),
                            "title": item.get("title"),
                            "abstract": item.get("abstract", "")[:200] + "..." if item.get("abstract") else "",
                            "evidence_type": "functional_program"
                        })
            
            # Context-specific query (if tissue/disease context provided)
            if tissue_context or disease_context:
                print(f"Batch querying PubMed for tissue/disease context ({len(genes)} genes)...")
                context_queries = genes.copy()
                if tissue_context:
                    context_queries.append(tissue_context)
                if disease_context:
                    context_queries.append(disease_context)
                result = self.pubmed_tool.execute(queries=context_queries, max_results=max_results * len(genes))
                if result.get("items"):
                    for item in result["items"]:
                        matched_genes = [g for g in genes if g.lower() in (item.get("title", "") + " " + item.get("abstract", "")).lower()]
                        for gene in matched_genes:
                            all_evidence["tissue_disease_context"].append({
                                "gene": gene,
                                "source": "PubMed",
                                "pmid": item.get("pmid"),
                                "url": item.get("url"),
                                "title": item.get("title"),
                                "abstract": item.get("abstract", "")[:200] + "..." if item.get("abstract") else "",
                                "evidence_type": "tissue_disease_context"
                            })
        
        # Remove duplicates based on (gene, pmid) pairs
        seen = set()
        for evidence_type in ["hard_lineage_markers", "functional_programs", "tissue_disease_context"]:
            unique_items = []
            for item in all_evidence[evidence_type]:
                key = (item["gene"], item.get("pmid", ""), item.get("source", ""))
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
            all_evidence[evidence_type] = unique_items
        
        print(f"Evidence collected: {len(all_evidence['hard_lineage_markers'])} lineage markers, "
              f"{len(all_evidence['functional_programs'])} functional programs, "
              f"{len(all_evidence['tissue_disease_context'])} context items")
        
        return all_evidence
    
    def _get_llm_gene_knowledge(self, genes: List[str], tissue_context: Optional[str] = None,
                               disease_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to quickly retrieve gene knowledge without PubMed queries.
        This is much faster for common genes.
        """
        try:
            llm_engine = ChatOpenAI(model_string=self.model_string, is_multimodal=False, api_key=self.api_key)
            
            prompt = f"""You are a gene annotation expert. For the following genes, provide their known functions, lineage markers, and cell type associations.

Genes: {', '.join(genes)}
Tissue context: {tissue_context or 'not specified'}
Disease context: {disease_context or 'not specified'}

For each gene, provide:
1. Known lineage markers or cell type associations (if any)
2. Primary biological functions
3. Relevant tissue/disease context (if applicable)

Output in JSON format:
{{
  "genes": [
    {{
      "gene": "GENE_NAME",
      "lineage_markers": ["marker1", "marker2"],
      "functions": ["function1", "function2"],
      "tissue_context": "relevant tissue info",
      "confidence": "high/medium/low"
    }}
  ]
}}

Only include information you are confident about. Use "N/A" if information is not available."""
            
            response = llm_engine(prompt)
            response_text = response.strip()
            
            # Parse JSON from response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            llm_data = json.loads(response_text)
            
            # Convert to evidence format
            evidence = {
                "hard_lineage_markers": [],
                "functional_programs": [],
                "tissue_disease_context": []
            }
            
            for gene_info in llm_data.get("genes", []):
                gene = gene_info.get("gene", "")
                if gene not in genes:
                    continue
                
                # Add lineage markers
                if gene_info.get("lineage_markers"):
                    for marker in gene_info["lineage_markers"]:
                        evidence["hard_lineage_markers"].append({
                            "gene": gene,
                            "source": "LLM Knowledge",
                            "pmid": None,
                            "url": None,
                            "title": f"{gene} as {marker} marker",
                            "abstract": f"Known lineage marker: {marker}",
                            "evidence_type": "hard_lineage_marker"
                        })
                
                # Add functions
                if gene_info.get("functions"):
                    for func in gene_info["functions"]:
                        evidence["functional_programs"].append({
                            "gene": gene,
                            "source": "LLM Knowledge",
                            "pmid": None,
                            "url": None,
                            "title": f"{gene} function: {func}",
                            "abstract": f"Known function: {func}",
                            "evidence_type": "functional_program"
                        })
                
                # Add tissue context
                if gene_info.get("tissue_context") and gene_info["tissue_context"] != "N/A":
                    evidence["tissue_disease_context"].append({
                        "gene": gene,
                        "source": "LLM Knowledge",
                        "pmid": None,
                        "url": None,
                        "title": f"{gene} in {gene_info['tissue_context']}",
                        "abstract": gene_info["tissue_context"],
                        "evidence_type": "tissue_disease_context"
                    })
            
            return evidence
            
        except Exception as e:
            print(f"Error getting LLM gene knowledge: {e}")
            return {
                "hard_lineage_markers": [],
                "functional_programs": [],
                "tissue_disease_context": []
            }

    def _calculate_confidence_breakdown(self, evidence: Dict[str, Any], 
                                      intrinsic_status: str) -> Dict[str, float]:
        """Calculate confidence score breakdown based on evidence."""
        breakdown = {
            "hard_lineage_marker": 0.0,
            "functional_program_coherence": 0.0,
            "tissue_disease_context": 0.0,
            "conflicting_signals_penalty": 0.0,
            "completeness_factor": 1.0,
            "intrinsic_factor": 1.0
        }
        
        # 1. Hard lineage marker evidence
        if evidence["hard_lineage_markers"]:
            # Quality based on number and relevance
            marker_count = len(evidence["hard_lineage_markers"])
            marker_quality = min(1.0, marker_count / 3.0)  # Normalize to 0-1
            breakdown["hard_lineage_marker"] = self.EVIDENCE_WEIGHTS["hard_lineage_marker"] * marker_quality
        
        # 2. Functional program coherence
        if evidence["functional_programs"]:
            # Coherence based on number of genes with functional evidence
            program_count = len(evidence["functional_programs"])
            coherence = min(1.0, program_count / 5.0)  # Normalize to 0-1
            breakdown["functional_program_coherence"] = self.EVIDENCE_WEIGHTS["functional_program_coherence"] * coherence
        
        # 3. Tissue/disease context
        if evidence["tissue_disease_context"]:
            context_count = len(evidence["tissue_disease_context"])
            context_match = min(1.0, context_count / 3.0)  # Normalize to 0-1
            breakdown["tissue_disease_context"] = self.EVIDENCE_WEIGHTS["tissue_disease_context"] * context_match
        
        # 4. Conflicting signals
        if evidence["conflicting_signals"]:
            conflict_count = len(evidence["conflicting_signals"])
            conflict_severity = min(1.0, conflict_count / 2.0)
            breakdown["conflicting_signals_penalty"] = self.EVIDENCE_WEIGHTS["conflicting_signals"] * conflict_severity
        
        # 5. Completeness factor
        evidence_layers = sum([
            1 if evidence["hard_lineage_markers"] else 0,
            1 if evidence["functional_programs"] else 0,
            1 if evidence["tissue_disease_context"] else 0
        ])
        if evidence_layers == 3:
            breakdown["completeness_factor"] = 1.0
        elif evidence_layers == 2:
            breakdown["completeness_factor"] = 0.85
        else:
            breakdown["completeness_factor"] = 0.70
        
        # 6. Intrinsic factor
        if intrinsic_status == "non-intrinsic":
            breakdown["intrinsic_factor"] = 0.9
        
        return breakdown

    def _calculate_final_confidence(self, breakdown: Dict[str, float]) -> float:
        """Calculate final confidence score with constraints."""
        base_score = (
            breakdown["hard_lineage_marker"] +
            breakdown["functional_program_coherence"] +
            breakdown["tissue_disease_context"] +
            breakdown["conflicting_signals_penalty"]
        )
        
        confidence = base_score * breakdown["completeness_factor"] * breakdown["intrinsic_factor"]
        
        # Apply upper bounds
        if breakdown["intrinsic_factor"] < 1.0:  # non-intrinsic
            confidence = min(0.75, confidence)
        else:  # intrinsic
            if breakdown["hard_lineage_marker"] > 0.3:  # Strong hard markers
                confidence = min(0.85, confidence)
            else:
                confidence = min(0.80, confidence)
        
        if breakdown["conflicting_signals_penalty"] < 0:
            confidence = min(0.70, confidence)
        
        if breakdown["completeness_factor"] < 0.8:
            confidence = min(0.60, confidence)
        
        # Ensure minimum confidence
        confidence = max(0.0, min(0.95, confidence))
        
        return round(confidence, 2)

    def _format_evidence_for_llm(self, evidence: Dict[str, Any]) -> str:
        """Format evidence for LLM prompt."""
        formatted = "**Evidence Collected:**\n\n"
        
        if evidence["hard_lineage_markers"]:
            formatted += "**Hard Lineage Markers:**\n"
            for item in evidence["hard_lineage_markers"][:5]:
                formatted += f"- {item['gene']}: {item['title']} (PMID: {item['pmid']})\n"
            formatted += "\n"
        
        if evidence["functional_programs"]:
            formatted += "**Functional Programs:**\n"
            for item in evidence["functional_programs"][:5]:
                formatted += f"- {item['gene']}: {item['title']} (PMID: {item['pmid']})\n"
            formatted += "\n"
        
        if evidence["tissue_disease_context"]:
            formatted += "**Tissue/Disease Context:**\n"
            for item in evidence["tissue_disease_context"][:5]:
                formatted += f"- {item['gene']}: {item['title']} (PMID: {item['pmid']})\n"
            formatted += "\n"
        
        if evidence["conflicting_signals"]:
            formatted += "**Conflicting Signals:**\n"
            for item in evidence["conflicting_signals"]:
                formatted += f"- {item}\n"
            formatted += "\n"
        
        return formatted

    def _generate_inference_prompt(self, cluster: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Generate LLM prompt for functional hypothesis inference."""
        genes = cluster.get("regulator_genes", [])
        tissue = cluster.get("tissue_context", "unknown")
        disease = cluster.get("disease_context", "unknown")
        target_lineage = cluster.get("target_lineage", "unknown")
        
        evidence_text = self._format_evidence_for_llm(evidence)
        
        prompt = f"""You are a cell cluster functional hypothesis expert. Follow these strict rules:

【Input Data】
- Cluster ID: {cluster.get('cluster_id', 'Unknown')}
- Top regulator genes: {', '.join(genes)}
- Tissue context: {tissue}
- Disease context: {disease}
- Target lineage: {target_lineage}

{evidence_text}

【Reasoning Rules】
1. Evidence Hierarchy (priority order):
   (i) Hard lineage markers
   (ii) Functional program coherence among top genes
   (iii) Tissue/disease context compatibility
   (iv) Conflicting or ambiguous signals
   Higher-priority evidence must not be overridden by lower-priority cues.

2. Mandatory First Step: Intrinsic vs Non-Intrinsic Decision
   - Check if cluster has multiple hard lineage markers from non-target lineage
   - If yes → classify as "non-intrinsic" (immune-like or stromal)
   - If no → proceed to functional state inference

3. Functional State Inference (only for potentially intrinsic clusters):
   - Infer functional states/programs (e.g., stress-associated, metabolic-adapted, interface-associated)
   - NOT cell identities, subtypes, or developmental origins
   - Use axis-based interpretation (functional axes, not discrete categories)

4. Annotation Language:
   - Use probabilistic phrasing: "associated with", "enriched for", "consistent with"
   - Avoid definitive claims, causal language, or novel cell-type naming

5. Confidence Scoring:
   - Range: 0-1 (reflects plausibility, not correctness)
   - Intrinsic clusters: usually ≤ 0.8
   - Strong hard markers: can reach 0.85
   - Conflicting signals: reduce confidence

6. Uncertainty Principle:
   - All annotations are hypothesis-generating
   - Prefer broad, low-confidence interpretations over specific claims when evidence is weak

【Output Format (JSON)】
{{
  "intrinsic_status": "intrinsic" or "non-intrinsic",
  "functional_state_annotation": "1-2 sentences describing functional state",
  "functional_class_group": "group_name" or null,
  "confidence_score": 0.0-1.0,
  "rationale": "One sentence referencing evidence types",
  "key_evidence_summary": "Brief summary of most important evidence"
}}

Output ONLY valid JSON, no additional text."""
        
        return prompt

    def execute(self, clusters: List[Dict[str, Any]], max_literature_per_gene: int = 5, 
                include_grouping: bool = True, model_string: Optional[str] = None, 
                api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute functional hypothesis generation for cell clusters.
        
        Args:
            clusters: List of cluster dictionaries
            max_literature_per_gene: Max PubMed results per gene
            include_grouping: Whether to perform functional class grouping
            model_string: LLM model (overrides default)
            api_key: API key (overrides default)
        """
        try:
            if not clusters or len(clusters) == 0:
                return {
                    "formatted_output": "**⚠️ Error:** No clusters provided.",
                    "error": "No clusters provided",
                    "results": []
                }
            
            # Use provided or default model
            llm_model = model_string or self.model_string
            llm_key = api_key or self.api_key
            
            llm_engine = ChatOpenAI(model_string=llm_model, is_multimodal=False, api_key=llm_key)
            
            all_results = []
            
            # Process each cluster
            for cluster in clusters:
                cluster_id = cluster.get("cluster_id", f"Cluster_{len(all_results) + 1}")
                genes = cluster.get("regulator_genes", [])
                
                if not genes:
                    all_results.append({
                        "cluster_id": cluster_id,
                        "error": "No regulator genes provided",
                        "formatted_output": f"**⚠️ {cluster_id}:** No regulator genes provided."
                    })
                    continue
                
                print(f"\nProcessing {cluster_id} with genes: {', '.join(genes)}")
                
                # Step 1: Collect evidence
                print(f"Collecting evidence for {cluster_id}...")
                evidence = self._collect_evidence(
                    genes=genes,
                    tissue_context=cluster.get("tissue_context"),
                    disease_context=cluster.get("disease_context"),
                    max_results=max_literature_per_gene
                )
                
                # Step 2: Generate hypothesis using LLM
                print(f"Generating hypothesis for {cluster_id}...")
                prompt = self._generate_inference_prompt(cluster, evidence)
                
                try:
                    response = llm_engine(prompt)
                    # Try to parse JSON from response
                    response_text = response.strip()
                    # Remove markdown code blocks if present
                    if response_text.startswith("```"):
                        response_text = response_text.split("```")[1]
                        if response_text.startswith("json"):
                            response_text = response_text[4:]
                    response_text = response_text.strip()
                    
                    hypothesis = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"Error parsing LLM response: {e}")
                    print(f"Response: {response_text[:500]}")
                    hypothesis = {
                        "intrinsic_status": "unknown",
                        "functional_state_annotation": "Unable to parse hypothesis from LLM response.",
                        "functional_class_group": None,
                        "confidence_score": 0.0,
                        "rationale": "Error in hypothesis generation",
                        "key_evidence_summary": "N/A"
                    }
                
                # Step 3: Calculate confidence breakdown
                breakdown = self._calculate_confidence_breakdown(
                    evidence=evidence,
                    intrinsic_status=hypothesis.get("intrinsic_status", "intrinsic")
                )
                
                # Step 4: Calculate final confidence
                final_confidence = self._calculate_final_confidence(breakdown)
                hypothesis["confidence_score"] = final_confidence
                hypothesis["confidence_breakdown"] = breakdown
                
                # Step 5: Format evidence list
                evidence_list = []
                for evidence_type in ["hard_lineage_markers", "functional_programs", "tissue_disease_context"]:
                    for item in evidence[evidence_type]:
                        evidence_list.append({
                            "type": evidence_type,
                            "gene": item["gene"],
                            "source": item["source"],
                            "pmid": item["pmid"],
                            "url": item["url"],
                            "title": item["title"],
                            "relevance": "high" if evidence_type == "hard_lineage_markers" else "medium"
                        })
                
                hypothesis["evidence"] = evidence_list
                hypothesis["cluster_id"] = cluster_id
                
                # Step 6: Format output
                formatted_output = self._format_cluster_output(hypothesis, evidence)
                hypothesis["formatted_output"] = formatted_output
                
                all_results.append(hypothesis)
            
            # Step 7: Perform functional class grouping if requested
            if include_grouping and len(all_results) > 1:
                grouping_result = self._perform_grouping(all_results)
                all_results = grouping_result
            
            # Format final output
            final_formatted = self._format_final_output(all_results)
            
            return {
                "formatted_output": final_formatted,
                "results": all_results,
                "count": len(all_results)
            }
            
        except Exception as e:
            error_msg = f"**⚠️ Error generating functional hypotheses:** {str(e)}"
            print(f"Error in Cell_Cluster_Functional_Hypothesis_Tool: {e}")
            import traceback
            traceback.print_exc()
            return {
                "formatted_output": error_msg,
                "error": str(e),
                "results": []
            }

    def _format_cluster_output(self, hypothesis: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Format individual cluster output."""
        output = f"## {hypothesis['cluster_id']} Functional Hypothesis\n\n"
        output += f"**Intrinsic Status:** {hypothesis.get('intrinsic_status', 'unknown').title()}\n\n"
        output += f"**Functional State:** {hypothesis.get('functional_state_annotation', 'N/A')}\n\n"
        
        if hypothesis.get('functional_class_group'):
            output += f"**Functional Class Group:** {hypothesis['functional_class_group']}\n\n"
        
        output += f"**Confidence Score:** {hypothesis.get('confidence_score', 0.0):.2f}\n\n"
        output += f"**Rationale:** {hypothesis.get('rationale', 'N/A')}\n\n"
        
        # Confidence breakdown
        breakdown = hypothesis.get('confidence_breakdown', {})
        output += "**Confidence Breakdown:**\n"
        output += f"- Hard lineage markers: {breakdown.get('hard_lineage_marker', 0.0):.2f}\n"
        output += f"- Functional program coherence: {breakdown.get('functional_program_coherence', 0.0):.2f}\n"
        output += f"- Tissue/disease context: {breakdown.get('tissue_disease_context', 0.0):.2f}\n"
        output += f"- Conflicting signals penalty: {breakdown.get('conflicting_signals_penalty', 0.0):.2f}\n\n"
        
        # Key evidence
        evidence_list = hypothesis.get('evidence', [])[:5]  # Top 5
        if evidence_list:
            output += "**Key Evidence:**\n"
            for idx, item in enumerate(evidence_list, 1):
                output += f"{idx}. {item['gene']} - {item['title']} "
                if item.get('pmid'):
                    output += f"(PMID: {item['pmid']}, {item['url']})\n"
                else:
                    output += "\n"
            output += "\n"
        
        return output

    def _format_final_output(self, results: List[Dict[str, Any]]) -> str:
        """Format final output for all clusters."""
        output = f"**📊 Functional Hypotheses for {len(results)} Cluster(s)**\n\n"
        output += "=" * 60 + "\n\n"
        
        for result in results:
            if result.get('formatted_output'):
                output += result['formatted_output']
                output += "\n" + "-" * 60 + "\n\n"
        
        return output

    def _perform_grouping(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform functional class grouping (simplified - can be enhanced with LLM)."""
        # Simple grouping based on similar functional states
        # This can be enhanced with LLM-based grouping
        groups = {}
        group_counter = 1
        
        for result in results:
            functional_state = result.get('functional_state_annotation', '')
            # Simple keyword-based grouping (can be improved)
            if 'stress' in functional_state.lower():
                group_name = f"stress_response_group_{group_counter}"
            elif 'metabolic' in functional_state.lower():
                group_name = f"metabolic_group_{group_counter}"
            elif 'immune' in functional_state.lower() or 'inflammatory' in functional_state.lower():
                group_name = f"immune_response_group_{group_counter}"
            else:
                group_name = None
            
            if group_name:
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(result['cluster_id'])
                result['functional_class_group'] = group_name
        
        return results

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command
    tool = Cell_Cluster_Functional_Hypothesis_Tool()
    
    # Example clusters
    clusters = [
        {
            "cluster_id": "Cluster_1",
            "regulator_genes": ["NKX2-1", "SFTPC"],
            "tissue_context": "lung",
            "disease_context": "adenocarcinoma",
            "target_lineage": "epithelial"
        }
    ]
    
    try:
        result = tool.execute(clusters=clusters, max_literature_per_gene=3)
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(result.get("formatted_output", "No output"))
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

