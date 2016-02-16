package ford.com;

public class Feature {

	String feature;
	int TF = 0;
	double IDF = 1;
	double weight;
	int docFrequency;

	// set feature weight
	public void setWeight() {
		weight = TF * IDF;
	}

	// get inverse term frequency
	public void setIDF(int collectionFrequency) {
		IDF = Math.log10(collectionFrequency / docFrequency);
	}
}
