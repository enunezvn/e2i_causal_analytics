// ============================================================================
// E2I SEMANTIC MEMORY GRAPH SCHEMA
// FalkorDB (RedisGraph-compatible) Schema for Entity Relationships
// Version: 1.0
// ============================================================================
// 
// This schema defines the semantic graph structure for storing facts and
// relationships in the E2I Agentic Memory system.
// 
// For Local Pilot: Can be implemented with NetworkX + pickle persistence
// For Production: FalkorDB or AWS Neptune
// ============================================================================

// ============================================================================
// NODE TYPES (Vertices)
// ============================================================================

// --- CORE BUSINESS ENTITIES ---

// Patient Node
// Represents an individual patient in the journey
// CREATE (:Patient {
//     patient_id: STRING,           -- Primary identifier (hashed)
//     patient_hash: STRING,         -- Alternative hash
//     journey_stage: STRING,        -- awareness, consideration, trial, adoption, adherence
//     risk_score: FLOAT,
//     journey_complexity: FLOAT,
//     region: STRING,               -- northeast, south, midwest, west
//     age_group: STRING,
//     insurance_type: STRING,
//     created_at: DATETIME,
//     updated_at: DATETIME
// })

// HCP Node
// Healthcare Professional
// CREATE (:HCP {
//     hcp_id: STRING,
//     npi: STRING,
//     name: STRING,                 -- Anonymized or full depending on context
//     specialty: STRING,
//     sub_specialty: STRING,
//     practice_type: STRING,
//     region: STRING,
//     priority_tier: INTEGER,       -- 1-5
//     decile: INTEGER,
//     adoption_category: STRING,    -- early_adopter, mainstream, late_adopter
//     digital_engagement_score: FLOAT,
//     influence_score: FLOAT,
//     created_at: DATETIME,
//     updated_at: DATETIME
// })

// Brand Node
// Drug brand
// CREATE (:Brand {
//     brand_id: STRING,
//     brand_name: STRING,           -- Remibrutinib, Fabhalta, Kisqali
//     drug_class: STRING,
//     therapeutic_area: STRING,
//     indications: [STRING],
//     created_at: DATETIME
// })

// Region Node
// Geographic region
// CREATE (:Region {
//     region_id: STRING,
//     region_name: STRING,          -- Northeast, South, Midwest, West
//     states: [STRING],
//     population_size: INTEGER,
//     hcp_count: INTEGER,
//     patient_count: INTEGER
// })

// --- ANALYTICS ENTITIES ---

// KPI Node
// Key Performance Indicator
// CREATE (:KPI {
//     kpi_id: STRING,
//     kpi_name: STRING,
//     kpi_category: STRING,         -- engagement, clinical, business, operational
//     workstream: STRING,           -- WS1, WS2, WS3
//     calculation_method: STRING,
//     target_value: FLOAT,
//     threshold_warning: FLOAT,
//     threshold_critical: FLOAT,
//     unit: STRING
// })

// CausalPath Node
// Represents a discovered causal relationship
// CREATE (:CausalPath {
//     path_id: STRING,
//     path_description: STRING,
//     effect_size: FLOAT,
//     confidence: FLOAT,
//     method_used: STRING,          -- DoWhy, EconML, CausalML
//     discovery_date: DATE,
//     validation_status: STRING
// })

// Trigger Node
// HCP engagement trigger
// CREATE (:Trigger {
//     trigger_id: STRING,
//     trigger_type: STRING,
//     trigger_category: STRING,
//     priority: STRING,
//     precision_score: FLOAT,
//     expected_impact: FLOAT
// })

// --- AGENT ENTITIES ---

// Agent Node
// E2I AI Agent
// CREATE (:Agent {
//     agent_name: STRING,           -- orchestrator, causal_impact, gap_analyzer, etc.
//     tier: INTEGER,                -- 1-5
//     purpose: STRING,
//     capabilities: [STRING],
//     avg_response_time_ms: FLOAT
// })

// --- GRAPHITI ENTITIES (Temporal Knowledge Graph) ---

// Episode Node
// Represents a knowledge episode (fact, observation, interaction) with temporal validity
// CREATE (:Episode {
//     episode_id: STRING,           -- Unique identifier
//     content: STRING,              -- Episode text content
//     content_summary: STRING,      -- Brief summary for display
//     source: STRING,               -- Agent name or 'user'
//     session_id: STRING,           -- Associated session ID
//     group_id: STRING,             -- For grouping related episodes
//     valid_at: DATETIME,           -- When this fact became true
//     invalid_at: DATETIME,         -- When this fact became false (NULL if still valid)
//     created_at: DATETIME,
//     metadata: JSON                -- Additional structured data
// })

// Community Node
// Represents a cluster of related entities in the graph
// CREATE (:Community {
//     community_id: STRING,         -- Unique identifier
//     name: STRING,                 -- Community name
//     summary: STRING,              -- Natural language summary of the community
//     member_count: INTEGER,        -- Number of entities in community
//     cohesion_score: FLOAT,        -- Measure of community tightness
//     created_at: DATETIME,
//     updated_at: DATETIME
// })

// --- TEMPORAL ENTITIES ---

// TimePeriod Node
// For temporal queries
// CREATE (:TimePeriod {
//     period_id: STRING,
//     period_type: STRING,          -- day, week, month, quarter
//     start_date: DATE,
//     end_date: DATE,
//     label: STRING                 -- "Q3 2025", "Week 48", etc.
// })

// ============================================================================
// RELATIONSHIP TYPES (Edges)
// ============================================================================

// --- PATIENT RELATIONSHIPS ---

// Patient treated by HCP
// (:Patient)-[:TREATED_BY {
//     first_visit_date: DATE,
//     last_visit_date: DATE,
//     visit_count: INTEGER,
//     is_primary_hcp: BOOLEAN
// }]->(:HCP)

// Patient prescribed Brand
// (:Patient)-[:PRESCRIBED {
//     prescription_date: DATE,
//     dosage: STRING,
//     duration_days: INTEGER,
//     is_first_line: BOOLEAN,
//     outcome: STRING
// }]->(:Brand)

// Patient in Region
// (:Patient)-[:LOCATED_IN]->(:Region)

// Patient journey stage transition
// (:Patient)-[:TRANSITIONED_TO {
//     transition_date: DATE,
//     from_stage: STRING,
//     trigger_event: STRING
// }]->(:Patient)  // Self-reference for state changes

// --- HCP RELATIONSHIPS ---

// HCP prescribes Brand
// (:HCP)-[:PRESCRIBES {
//     volume_monthly: INTEGER,
//     market_share: FLOAT,
//     adoption_date: DATE,
//     preference_rank: INTEGER
// }]->(:Brand)

// HCP located in Region
// (:HCP)-[:PRACTICES_IN {
//     is_primary_location: BOOLEAN
// }]->(:Region)

// HCP influences HCP (peer network)
// (:HCP)-[:INFLUENCES {
//     influence_strength: FLOAT,
//     interaction_count: INTEGER,
//     network_type: STRING          -- referral, academic, social
// }]->(:HCP)

// HCP received Trigger
// (:HCP)-[:RECEIVED {
//     delivery_date: DATETIME,
//     channel: STRING,
//     accepted: BOOLEAN,
//     action_taken: BOOLEAN
// }]->(:Trigger)

// --- CAUSAL RELATIONSHIPS ---

// Causal path connects entities
// (:Entity)-[:CAUSES {
//     effect_size: FLOAT,
//     confidence: FLOAT,
//     time_lag_days: INTEGER,
//     mediators: [STRING],
//     confounders_controlled: [STRING]
// }]->(:Entity)

// KPI impacted by CausalPath
// (:CausalPath)-[:IMPACTS {
//     impact_magnitude: FLOAT,
//     direction: STRING             -- positive, negative
// }]->(:KPI)

// --- AGENT RELATIONSHIPS ---

// Agent analyzes Entity
// (:Agent)-[:ANALYZES {
//     last_analysis_date: DATETIME,
//     analysis_count: INTEGER
// }]->(:Entity)

// Agent discovered CausalPath
// (:Agent)-[:DISCOVERED {
//     discovery_date: DATETIME,
//     method_used: STRING
// }]->(:CausalPath)

// Agent generated Trigger
// (:Agent)-[:GENERATED]->(:Trigger)

// --- GRAPHITI RELATIONSHIPS ---

// Episode mentions Entity
// Used to connect knowledge episodes to the entities they reference
// (:Episode)-[:MENTIONS {
//     confidence: FLOAT,            -- Extraction confidence (0-1)
//     mention_type: STRING,         -- subject, object, context
//     extracted_at: DATETIME
// }]->(:Entity)  // Can point to any entity type

// Entity member of Community
// Links entities to their detected communities
// (:Entity)-[:MEMBER_OF {
//     membership_score: FLOAT,      -- Strength of membership (0-1)
//     joined_at: DATETIME,
//     role: STRING                  -- hub, peripheral, bridge
// }]->(:Community)

// Episode relates to Episode
// Temporal relationship between episodes
// (:Episode)-[:FOLLOWS {
//     time_gap_seconds: INTEGER,
//     same_session: BOOLEAN
// }]->(:Episode)

// Episode invalidates Episode
// When new information supersedes old
// (:Episode)-[:INVALIDATES {
//     invalidation_reason: STRING,
//     invalidated_at: DATETIME
// }]->(:Episode)

// --- TEMPORAL RELATIONSHIPS ---

// Entity measured in TimePeriod
// (:KPI)-[:MEASURED_IN {
//     value: FLOAT,
//     baseline: FLOAT,
//     target: FLOAT
// }]->(:TimePeriod)

// ============================================================================
// INDEXES FOR PERFORMANCE
// ============================================================================

// Node indexes
// CREATE INDEX ON :Patient(patient_id)
// CREATE INDEX ON :Patient(region)
// CREATE INDEX ON :Patient(journey_stage)
// CREATE INDEX ON :HCP(hcp_id)
// CREATE INDEX ON :HCP(npi)
// CREATE INDEX ON :HCP(specialty)
// CREATE INDEX ON :HCP(region)
// CREATE INDEX ON :Brand(brand_name)
// CREATE INDEX ON :KPI(kpi_name)
// CREATE INDEX ON :CausalPath(path_id)
// CREATE INDEX ON :Agent(agent_name)
// CREATE INDEX ON :TimePeriod(period_type, start_date)

// Graphiti indexes
// CREATE INDEX ON :Episode(episode_id)
// CREATE INDEX ON :Episode(session_id)
// CREATE INDEX ON :Episode(source)
// CREATE INDEX ON :Episode(valid_at)
// CREATE INDEX ON :Community(community_id)
// CREATE INDEX ON :Community(name)

// Full-text indexes for search
// CREATE FULLTEXT INDEX patient_search FOR (n:Patient) ON EACH [n.patient_id, n.region]
// CREATE FULLTEXT INDEX hcp_search FOR (n:HCP) ON EACH [n.name, n.specialty, n.region]
// CREATE FULLTEXT INDEX episode_search FOR (n:Episode) ON EACH [n.content, n.content_summary]

// ============================================================================
// EXAMPLE QUERIES
// ============================================================================

// 1. Find causal paths from HCP engagement to patient outcomes
// MATCH path = (h:HCP)-[:RECEIVED]->(t:Trigger)-[:CAUSES*1..3]->(outcome:KPI)
// WHERE outcome.kpi_category = 'clinical'
// RETURN path, [r IN relationships(path) | r.effect_size] AS effects

// 2. Get HCP influence network in a region
// MATCH (h1:HCP)-[i:INFLUENCES]->(h2:HCP)
// WHERE h1.region = 'northeast' AND i.influence_strength > 0.5
// RETURN h1.name, h2.name, i.influence_strength
// ORDER BY i.influence_strength DESC

// 3. Trace a patient's treatment journey
// MATCH journey = (p:Patient {patient_id: $pid})-[:TREATED_BY|PRESCRIBED*]-()
// RETURN journey

// 4. Find what agents discovered specific causal relationships
// MATCH (a:Agent)-[:DISCOVERED]->(c:CausalPath)-[:IMPACTS]->(k:KPI)
// WHERE k.kpi_name = 'trigger_acceptance_rate'
// RETURN a.agent_name, c.path_description, c.effect_size

// 5. Multi-hop investigation: Why did Kisqali adoption increase?
// MATCH (brand:Brand {brand_name: 'Kisqali'})<-[:PRESCRIBES]-(h:HCP)
//       <-[:RECEIVED]-(t:Trigger)<-[:GENERATED]-(a:Agent)
// WHERE t.accepted = true
// WITH brand, h, t, a
// MATCH (t)-[:CAUSES*1..3]->(impact:KPI)
// RETURN a.agent_name, t.trigger_type, h.specialty, impact.kpi_name

// --- GRAPHITI QUERIES ---

// 6. Find all episodes from a specific session
// MATCH (e:Episode {session_id: $session_id})
// WHERE e.valid_at IS NOT NULL AND e.invalid_at IS NULL
// RETURN e ORDER BY e.created_at DESC

// 7. Get entities mentioned in recent episodes
// MATCH (e:Episode)-[m:MENTIONS]->(entity)
// WHERE e.created_at > datetime() - duration('P7D')
// RETURN entity, COUNT(e) AS mention_count, AVG(m.confidence) AS avg_confidence
// ORDER BY mention_count DESC

// 8. Find related entities through shared community membership
// MATCH (e1:Entity)-[:MEMBER_OF]->(c:Community)<-[:MEMBER_OF]-(e2:Entity)
// WHERE e1 <> e2 AND e1.id = $entity_id
// RETURN e2, c.name AS community, c.summary
// ORDER BY c.cohesion_score DESC

// 9. Trace knowledge evolution (fact supersession)
// MATCH path = (old:Episode)<-[:INVALIDATES*1..3]-(new:Episode)
// WHERE old.episode_id = $episode_id
// RETURN path, [e IN nodes(path) | e.content_summary] AS evolution

// 10. Find agent-generated knowledge about a topic
// MATCH (a:Agent)-[:GENERATED|DISCOVERED]->(artifact)
// MATCH (e:Episode)-[:MENTIONS]->(artifact)
// WHERE e.content CONTAINS $keyword
// RETURN a.agent_name, artifact, e.content_summary

// ============================================================================
// NETWORKX EQUIVALENT STRUCTURE (for local pilot)
// ============================================================================
// 
// import networkx as nx
// import pickle
// 
// # Create directed graph
// G = nx.DiGraph()
// 
// # Add nodes with type attribute
// G.add_node('patient_001', node_type='Patient', region='northeast', risk_score=0.7)
// G.add_node('hcp_001', node_type='HCP', specialty='Oncology', tier=1)
// G.add_node('Kisqali', node_type='Brand', therapeutic_area='Oncology')
// 
// # Add edges with relationship type
// G.add_edge('patient_001', 'hcp_001', 
//            rel_type='TREATED_BY', 
//            first_visit='2025-01-15',
//            visit_count=5)
// 
// G.add_edge('hcp_001', 'Kisqali',
//            rel_type='PRESCRIBES',
//            volume_monthly=25,
//            market_share=0.35)
// 
// # Persist to disk
// with open('semantic_memory.pkl', 'wb') as f:
//     pickle.dump(G, f)
// 
// # Query example: Find all HCPs who prescribe Kisqali
// prescribers = [n for n in G.predecessors('Kisqali') 
//                if G.nodes[n].get('node_type') == 'HCP']
// ============================================================================

// ============================================================================
// DATA SYNC STRATEGY
// ============================================================================
// 
// The semantic graph is the source of truth for entity relationships.
// A sync process periodically updates the semantic_memory_cache table in
// PostgreSQL for fast retrieval during investigation hops.
// 
// Sync Process:
// 1. Query FalkorDB for recently updated relationships
// 2. Transform to triplet format (subject, predicate, object)
// 3. Upsert into semantic_memory_cache with embeddings
// 4. Mark stale cache entries as invalid (valid_until = NOW())
// 
// Sync Frequency: Every 5 minutes for hot data, hourly for full sync
// ============================================================================
