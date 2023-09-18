package elki.datasource.filter.selection;

import elki.data.LabelList;
import elki.datasource.bundle.BundleMeta;
import elki.datasource.filter.AbstractStreamFilter;
import elki.datasource.filter.FilterUtil;
import elki.logging.Logging;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.PatternParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SampleClassFilter extends AbstractStreamFilter {

    private static final Logging LOG = Logging.getLogger(SampleClassFilter.class);

    private final Matcher labelMatcher;

    private int labelColumn = -1;
    private double acceptProbability;
    private Random random;

    public SampleClassFilter(Pattern labelPattern, double prob, RandomFactory rnd) {
        super();
        this.labelMatcher = labelPattern.matcher("");
        this.acceptProbability = prob;
        this.random = rnd.getSingleThreadedRandom();
    }

    @Override
    public BundleMeta getMeta() {
        return source.getMeta();
    }

    @Override
    public Object data(int rnum) {
        return source.data(rnum);
    }

    @Override
    public Event nextEvent() {
        while (true){
            Event ev = source.nextEvent();

            switch(ev) {
                case END_OF_STREAM:
                    return ev;
                case META_CHANGED:
                    if(labelColumn < 0) {
                        BundleMeta meta = source.getMeta();
                        labelColumn = FilterUtil.findLabelColumn(meta);
                    }
                    return ev;
                case NEXT_OBJECT:

                    if(labelColumn > 0){
                        Object label = source.data(labelColumn);
                        if(label instanceof LabelList) {
                            boolean match = false;
                            final LabelList ll = (LabelList) label;
                            for(int i = 0; i < ll.size(); i++) {
                                labelMatcher.reset(ll.get(i));
                                if(labelMatcher.matches()) {
                                    match = true;
                                    break;
                                }
                            }

                            if (match){
                                if (random.nextDouble() > acceptProbability) {
                                    continue;
                                }
                            }
                        }
                        else {
                            labelMatcher.reset(label.toString());
                            if(labelMatcher.matches()) {
                                if (random.nextDouble() > acceptProbability) {
                                    continue;
                                }
                            }
                        }
                        return ev;
                    }  //else No labels known
            }
        }
    }

    public static class Par implements Parameterizer {

        public static OptionID PROB_ID = new OptionID("labelsampling.p", "Sampling probability for given class.");
        public static OptionID SEED_ID = new OptionID("labelsampling.seed", "Rndom generator seed for sampling");
        public static OptionID LABEL_PATTERN_ID = new OptionID("labelsampling.pattern", "The filter pattern to sample.");
        protected Pattern pattern;
        protected double prob;
        protected RandomFactory rnd;
        @Override
        public void configure(Parameterization config) {
            new PatternParameter(LABEL_PATTERN_ID).grab(config, x -> pattern = x);
            new DoubleParameter(PROB_ID).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE) //
                    .addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE) //
                    .grab(config, x -> prob = x);
            new RandomParameter(SEED_ID).grab(config, x -> rnd = x);
        }

        @Override
        public Object make() {
            return new SampleClassFilter(pattern, prob, rnd);
        }
    }
}
