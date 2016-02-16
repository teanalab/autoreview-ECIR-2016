package ford.com;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Vector;

import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class UtilityClass {

	public static Instances getData(Vector<Entry> train, String trainFolder, String dataFile, String splitOptions, boolean willUseLIWC) throws Exception{
		
		// store documents and docsId for future use
		Vector<String> documents = new Vector<String>();
		Vector<Integer> docIDs = new Vector<Integer>();
		
		// create an object of file manager
		FileManager fmgr = new FileManager();

		// prepare train data set
		fmgr.createDataFolder(train, trainFolder);
		TextDirectoryLoader loader = new TextDirectoryLoader();
		loader.setDirectory(new File(trainFolder));
		Instances dataRaw = loader.getDataSet();
		StringToWordVector filter = new StringToWordVector();
		filter.setOptions(weka.core.Utils.splitOptions(splitOptions));
		filter.setInputFormat(dataRaw);
		Instances trainFiltered = Filter.useFilter(dataRaw, filter);
		Reorder reorder = new Reorder();
		reorder.setOptions(weka.core.Utils.splitOptions("-R 2-last,first"));
		reorder.setInputFormat(trainFiltered);
		trainFiltered = Filter.useFilter(trainFiltered, reorder);
		
		// generate attribute file format for weka
		fmgr.setDocuments(documents, docIDs);
		// last parameter determine that whether we use LIWC or not?
		fmgr.makeWekaFileFormat(trainFiltered, "reviews", documents, docIDs, dataFile, willUseLIWC);
		
		// Load arff file
		BufferedReader reader = new BufferedReader(new FileReader("liwc/reviews.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		
		return data;
	}
}
