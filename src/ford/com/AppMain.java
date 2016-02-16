package ford.com;

import java.io.File;
import java.util.Vector;

public class AppMain {

	public static void main(String[] args) throws Exception {

		// Create the object of file manager
		FileManager fm = new FileManager();
				
		// Set temporary container for training data
		String trainFolder = "train";
		
		// set parameter for LIWC feature
		boolean willUseLIWC = false;
		
		// Refresh train folder
		fm.removeDirectory(new File(trainFolder));
		File trainNewFolder = new File(trainFolder);
		trainNewFolder.mkdir();
		
		// Specify the location of data to evaluate the performance of models
		String dataContainer = "data";
		String dataFile = "pos-neg-objective.txt";
		
		/*********************************
			Create gold standard       
		*******************************/
		//fm.saveReviewWithModel("kbb");
		//fm.saveReviewWithModel("edmund");
		
		/********************************************************************
			Divide data into different files based on categorical hierarchy       
		********************************************************************/
		//fm.writeAutoReviewHierarchy("rawdata/msn_annotated_sample.txt");
		//fm.writeImportantAutoReview("rawdata/msn_annotated_sample.txt");
		
		// Read data from given file
		Vector<Entry> inputData = fm.readRawData(dataContainer + "/" + dataFile);
		
		/********************************************************************
		    Done POS tagging and some analysis for research paper
	    ********************************************************************/
		//NLPClass nlp = new NLPClass();
		//nlp.doPOSTagging("pos-neg-objective");
		//nlp.generatePatterns("pos-neg-objective");		
		//nlp.displayPOSTagOnly();		
		
		
		/********************************************************************
	    	Call models for evaluate their performance
        ********************************************************************/
		// Naive bayes model
		NaiveBayesClassifier nbClassifier = new NaiveBayesClassifier();
		nbClassifier.testAndRunClassifier(inputData, trainFolder, dataFile, willUseLIWC);

		// Logistic regression model
		LibLinearClassifier liblinear = new LibLinearClassifier();
		liblinear.testAndRunClassifier(inputData, trainFolder, dataFile, willUseLIWC);
		
		// Weka SVM model
		SVMClassifier svmCls = new SVMClassifier();
		svmCls.testAndRunClassifier(inputData, trainFolder, dataFile, willUseLIWC);
		
		// Refresh train folder
		fm.removeDirectory(new File(trainFolder));
	}

}
