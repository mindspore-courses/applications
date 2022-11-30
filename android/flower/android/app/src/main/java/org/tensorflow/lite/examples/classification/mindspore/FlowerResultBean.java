package org.tensorflow.lite.examples.classification.mindspore;

public class FlowerResultBean {
    public static final String[] FLOWER_LABEL ={"daisy", "dandelion", "roses", "sunflowers", "tulips"};

    private String label;
    private float score;

    public FlowerResultBean(int labelIndex,float score){
        this.label = FLOWER_LABEL[labelIndex];
        this.score = score;
    }

    public String getLabel() {
        return label;
    }
}
