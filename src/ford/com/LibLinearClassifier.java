package ford.com;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.*;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instances;

public class LibLinearClassifier {
	
	// run and train the classifier, finally test the classifier using test dataset
	public void testAndRunClassifier(Vector<Entry> train, String trainFolder, 
			String inputMapFile, boolean willUseLIWC) throws Exception {
		
		// get data
		Instances data = UtilityClass.getData(train, trainFolder, "-C", inputMapFile.replace(".txt", ""), willUseLIWC);
		
		// Build the Logistic regression model
		Classifier classifier = new LibLINEAR();
		classifier.buildClassifier(data);
		// set model parameters
		String[] options = weka.core.Utils.splitOptions("-S 0 -C 0.1");
		classifier.setOptions(options);
		
		// use 10 folds and store the results in report folder
		Evaluation eval = new Evaluation(data);
		int folds = 10, seed = 1;
		Random rand = new Random(seed); 
		eval.crossValidateModel(classifier, data, folds, rand);
		System.out.println("\n==== LIBSVM MODEL ====\n");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
		// write results to file
		PrintWriter pw = new PrintWriter(new FileWriter(("report/" + inputMapFile), true));
		pw.println("\n\n==== L2 regularized logistic regression model ====\n\n");
		pw.println("==== Classification Summary ====");
		pw.println(eval.toMatrixString());
		pw.println(eval.toSummaryString());
		pw.println(eval.toClassDetailsString());
		pw.close();

	}
}
