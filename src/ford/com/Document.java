package ford.com;

import java.util.HashMap;
import java.util.Map;

public class Document {
	String name, label, text;
	Map<String, Feature> featureMap = new HashMap<String, Feature>();
	int length = 0;

	// constructor with document name
	public Document(String docName) {
		name = docName;
	}

	// default constructor
	public Document() {

	}

	// extract features from document
	public void createFeatures(String doc, Map<String, Integer> wordCollection) {
		Feature featureObj;
		String[] tokens;

		// set label and corresponding text
		this.label = doc.substring(0, 1).trim();
		this.text = doc.substring(2).trim();
		this.text = this.text.replace("?", " ?");
		this.text = this.text.replace("!", " !");

		// get tokens from the text
		tokens = this.text.replace("\t", " ").split(" ");
		length = tokens.length;

		// set term frequency of each feature
		for (int i = 0; i < length; i++) {
			String currentToken = getFreshTerm(tokens[i]);
			int tf = getTF(tokens, currentToken);
			if (tf != 0 && !currentToken.equals("")) {
				featureObj = new Feature();
				featureObj.TF = tf;
				featureObj.feature = currentToken;
				featureMap.put(currentToken, featureObj);
				wordCollection.put(currentToken, 0);
			}
		}

	}

	// get term frequency of a term
	public int getTF(String[] terms, String term) {
		int termFrequency = 0;
		term = getFreshTerm(term);
		for (int i = 0; i < terms.length; i++) {
			terms[i] = getFreshTerm(terms[i]);
			if (terms[i].equals(term)) {
				termFrequency++;
				terms[i] = "";
			}
		}
		return termFrequency;
	}

	// get characters only for word
	public String getFreshTerm(String term) {
		term = term.replace(".", "");
		term = term.replace(",", "");
		term = term.replace(")", "");
		term = term.replace("(", "");
		term = term.replace("]", "");
		term = term.replace("[", "");
		term = term.replace("\"", "");
		term = term.replace("n't", "n");
		term = term.toLowerCase();
		return term;
	}
}
