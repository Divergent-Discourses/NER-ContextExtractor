# Tibetan NER Training Dataset Creation Using Naïve Context-Searching: A Review
## Ingredients
Entity Lists:
-	BDRC Person Names
-	BDRC Place Names
-	BDRC Publisher Names
-	Harvard 2009 Tibetan Towns
-	TBRC Place Names
-	Tibetan NER Training Data (Hill et al., 2017)
-	Map Index by G Verhufen

Class Label Selections:
-	PERSON
-	PLACE
-	PUBLISHER
  
Corpus for Context Extraction:
-	Online Tibetan-language news articles (set of normalised .txt files)


## Background
Named Entity Recognition (NER) is the computational task of automatically identifying entities within a text and assigning them to predefined class labels, such as person, place, or publisher. NER can be used to transform unstructured textual data into structured datasets. For example, it allows for the creation of structured lists of people mentioned in news articles or enables corpus search functionalities to retrieve all articles referencing a specific person or place.
The Divergent Discourses project aimed to train an NER model to annotate a historical newspaper corpus (1955-1962) with person, place, and publisher labels to facilitate computational analysis. To accomplish this, we needed a dataset containing in-context examples of these entity types. For example:
•	**Entity**: ངག་དབང་ཆོས་འཛིན
•	**Class Label**: PERSON
•	**Context**: དེ་ལས་བདག་ལ་སྤོབས་པ་ཆེན་པོ་སྐྱེས་བྱུང་ལ་དགོན་སྡེ་ཡོངས་ཀྱི་དགེ་འདུན་པའི་གཟི་བརྗིད་ཀྱང་རེད" མི་ལོ་དྲུག་ཅུར་ཉེ་བའི་   **[ངག་དབང་ཆོས་འཛིན་]**   གྲྭ་པ་བྱས་ནས་ལོ་40སོང་ཞིང་།

This dataset would serve as training data for our NER model.

For clarity, we define key terms as follows:

- **Contexts**: Sentences/sub-sentences in which an entity occurs.
- **Entity**: A unique proper noun to be labelled (e.g., ཞི་ཅིན་ཕིང).
- **Entity Classes**: The labels assigned to entities (e.g., PERSON, PLACE, PUBLISHER).


## Method
We compiled a list of 56,548 Tibetan-language place, person, and publisher entities and selected a modern newspaper corpus to extract examples of these entities in context. This corpus was chosen for its linguistic similarity to the historical newspaper corpus we intended to annotate.

We then developed a Python script to iterate through the newspaper articles, searching for occurrences of the entities. Upon identifying an entity, we extracted the surrounding text from the last-occurring shad ( ། ) preceding the entity to the next-occurring shad following it. We aimed to extract five contextual examples per entity and deemed an entity ‘finalised’ once five examples had been retrieved.

This approach deviates from the standard method of dataset creation for NER, which typically involves manual annotation. Manual annotation ensures high accuracy and maximises the utility of the corpus used for context extraction - an especially important consideration in low-resource language contexts, where available textual data is limited, and even less is available in the specific genre or tone required for the task. However, given the resource-intensive nature of human annotation, we experimented with this automated method as an alternative.


## Results and Discussion
A qualitative review of intermediary results suggested that our approach produced an error-prone dataset, which was unlikely to facilitate effective model training.
We already possessed a manually annotated dataset (Hill et al., 2017) which we had used to train an initial experimental model. However, our results were unsatisfactory: the model extracted only **7.01%** of entities from **13,908 entity contexts**, and of these, **51.79% were correct**. This meant that **only 3.63% of entities were correctly extracted**. To improve the model’s tagging accuracy, we attempted to generate a supplementary dataset automatically. However, after training an NER model on the combined dataset - incorporating both our intermediary dataset and our original manually annotated dataset - the model's performance remained unchanged, suggesting that the supplementary dataset was ineffective.

To further evaluate our dataset, we conducted a **qualitative review** of a randomly sampled subset:

1.	**60 entities** from our entire entity set of 56,548 entities, and
2.	**16 entities** from our set of 2,873 ‘finalised’ entities (i.e., entities for which our method extracted 5 contextual examples).

Our review led to the following observations:


## 1. Low Extraction Rate
- Of the 60 randomly sampled entities, our code retrieved contextual examples for only 6 entities when we halted the process after several days.
- This suggests a 10% extraction rate (though the rate would have been slightly higher had we allowed the code to complete execution).


## 2. Low Accuracy
-	Of the six retrieved entities, only three had fully correct contexts.
-	One entity had partially correct contexts.
-	Two entities had entirely incorrect contexts.


## 3.  Low Retrieval Rate for Certain Entity Types
- Of the six retrieved entities:
  - Four were labeled as ‘PLACE’
  - Two were labeled as ‘PERSON’
  - None were labeled as ‘PUBLISHER’

- This suggests a mismatch between the entity types we aimed to extract and the corpus we used for context retrieval.
- While our corpus may have been a good match in terms of stylistic tone, it does not appear to have been well-matched in terms of historical period or content.
- Several of our entity lists were compiled by the BDRC (Buddhist Digital Resource Center), which primarily archives historical and religious texts. As a result:
  - Person and publisher names from these lists were less likely to appear in our extraction corpus.
  - Place names were more likely to be retrieved, as geographic names tend to remain stable over time.
    
To gain a more comprehensive understanding of retrieval quality when contexts were available, we also reviewed contexts from our set of ‘finalised’ entities’.

Our key findings were as follows.


## 4. Shorter entities were more error-prone
Our code naïvely searched for exact string matches of an entity within the text, leading to errors where an entity appeared as part of a longer, unrelated word.

### Example of Incorrect Retrieval

- **Entity**: རྒྱ་མ [PLACE]
- **Incorrect Context**: གཏན་འདུན་གྱི་དམིགས་བྱ་གཅིག་པའི་མཐའ་ཡས་རྒྱ་མཚོ་ཆེ་རུ་ཆས་འདོད།
- **Issue**: རྒྱ་མ was incorrectly matched within རྒྱ་མཚོ ("ocean"), which is not the intended entity.

### Proposed Solution
A more robust approach would involve searching only for entity occurrences followed by a tsheg (་), indicating the end of a word or syllable. However, we opted against this approach because:

-	Tibetan words often have case particles attached.
-	Our NER model needed to be trained to recognise entities with these markers attached, as they frequently appear in running text.

Instead, we created a modified retrieval method for shorter entities (≤3 syllables) which:

- Tokenised the corpus text to separate word stems from case markers.
- Searched for tokens instead of raw string matches.

However, this approach proved ineffective for longer entities, as our tokenisation method tended to split multi-word entities into smaller components.

### Example of Correct Retrieval (Longer Entity)
- **Entity**: ཧི་མ་ལ་ཡའི་རི་རྒྱུད་ [PLACE]
- **Correct Context**: མཁའ་ཡི་ཉི་མ་འདྲ་བའི་སྐྱེས་ཆེན་གང་དེའི་ཞབས་རྗེས་ཀྱི་གཟི་འོད་ཧི་མ་ལ་ཡའི་རི་རྒྱུད་གང་སར་སེར་ལྷམ་མེར་འཕྲོས་ནས་འདུག་ཅིང་།
- **Issue Avoided**: Longer entities tended to be correctly retrieved, as they were less likely to appear as substrings in unrelated words.


## 5. The Homonym Problem
Many Tibetan person and place names have alternative meanings, leading to incorrect matches.

### Example of Incorrect Retrieval Due to Homonymy
- **Entity**: འཛོམས་པ་ [PERSON]
- **Incorrect Context**: ཟླ་མཇུག་ནས་ཉི་སངས་ཟླ་གསུམ་གདོང་ལྔའི་ཁྱིམ་དུ་འཛོམས་པས་ཤར་ལྷོའི་རྒྱུད་དུ་ཆར་ཆུ་ཆེ་ཞིང་ཐོག་སེར་རྒོད།
- **Issue**: Here, འཛོམས་པ means ‘gathered’ rather than a person's name.

### Proposed Solution
A vector-based retrieval approach that accounts for semantic context (e.g., word embeddings) would better handle lexical ambiguity.


## 6. Partial Name Retrieval
Person and place names often appeared in correctly retrieved contexts, but incomplete entity spans were extracted because they were partial names appearing within longer names.

### Example of Partial Retrieval
- **Entity**: ཕུན་ཚོགས་ [PERSON]
- **Retrieved Context**: བསྟན་འཛིན་ཕུན་ཚོགས་ཀྱིས།
- **Issue**: The full name is བསྟན་འཛིན་ཕུན་ཚོགས་, but we only labelled the entity span for ཕུན་ཚོགས་. The entity span for the full name should be labelled to teach the model to tag entire entities rather than entity chunks.


## Conclusion
Our findings motivate the use of large language model (LLM)-based tagging, provided the selected LLM adequately represents Tibetan in its multilingual vector space.
While LLM-generated annotations will not be perfect, they are likely to:

- Surpass our current context-extraction method in accuracy.
- Resolve issues related to lexical ambiguity.
- Extract a wider range of entities, as LLMs can tag multiple entities in a given text chunk, rather than being limited to predefined entity lists.

## References
Hill, Nathan W., & Garrett, Edward. (2017). A part-of-speech (POS) lexicon of Classical Tibetan for NLP [Data set]. Zenodo. http://doi.org/10.5281/zenodo.574876 


# Usage
Inputs:
- **Dataframe** formatted as in sample_entity_df.**csv** listing entities.
- **Corpus** of one or several **.txt** files from which you wish to abstract your listed entities.

Outputs:
- entity_dict (json) - lists entities which have not yet been finalised alongside their 5 context slots. 
- finalisation_order (json) - lists the order in which entities become finalised (i.e. 5 contextual examples have been extracted from the provided corpus)
- finalised_entities (json) - lists finalised entities alongside their 5 extracted contextual examples
- context_sources (json) - lists each entity alongside the .txt file name in which each extracted context was found.

**Before running, open entity_context_finder.py and modify the listed input/output filepaths (lines 291-296).** 
