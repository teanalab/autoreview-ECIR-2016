package ford.com;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;

public class NaiveBayesClassifier {

	public void testAndRunClassifier(Vector<Entry> train, String trainFolder, String inputMapFile, boolean willUseLIWC)
			throws Exception {

		// get data
		Instances data = UtilityClass.getData(train, trainFolder, "-C", inputMapFile.replace(".txt", ""), willUseLIWC);
		
		// Build model - NAIVE BAYES multinomial
		Classifier classifier = new NaiveBayesMultinomial();
		classifier.buildClassifier(data);
				
		// use 10 folds and store the results in report folder
		Evaluation eval = new Evaluation(data);
		int folds = 10, seed = 1;
		Random rand = new Random(seed); 
		eval.crossValidateModel(classifier, data, folds, rand);
		System.out.println("\n==== NAIVE BAYES MODEL ====\n");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
		// write results to file
		PrintWriter pw = new PrintWriter(new FileWriter(("report/" + inputMapFile), true));
		pw.println("\n\n==== NAIVE BAYES MODEL ====\n\n");
		pw.println("==== Classification Summary ====");
		pw.println(eval.toSummaryString());
		pw.println(eval.toClassDetailsString());
		pw.close();

	}

}
