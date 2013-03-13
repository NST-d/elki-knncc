package de.lmu.ifi.dbs.elki.index.tree.spatial.rstarvariants.rdknn;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2013
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.distance.distancefunction.SpatialPrimitiveDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancefunction.minkowski.EuclideanDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.NumberDistance;
import de.lmu.ifi.dbs.elki.index.tree.spatial.rstarvariants.AbstractRStarTreeFactory;
import de.lmu.ifi.dbs.elki.index.tree.spatial.rstarvariants.AbstractRTreeSettings;
import de.lmu.ifi.dbs.elki.persistent.PageFile;
import de.lmu.ifi.dbs.elki.persistent.PageFileFactory;
import de.lmu.ifi.dbs.elki.utilities.ClassGenericsUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.GreaterConstraint;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameterization.Parameterization;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.IntParameter;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.ObjectParameter;

/**
 * Factory for RdKNN R*-Trees.
 * 
 * @author Erich Schubert
 * 
 * @apiviz.stereotype factory
 * @apiviz.uses RdKNNTreeIndex oneway - - «create»
 * 
 * @param <O> Object type
 */
public class RdKNNTreeFactory<O extends NumberVector<?>, D extends NumberDistance<D, ?>> extends AbstractRStarTreeFactory<O, RdKNNNode<D>, RdKNNEntry<D>, RdKNNTree<O, D>, AbstractRTreeSettings> {
  /**
   * Parameter for k
   */
  public static final OptionID K_ID = new OptionID("rdknn.k", "positive integer specifying the maximal number k of reverse " + "k nearest neighbors to be supported.");

  /**
   * The default distance function.
   */
  public static final Class<?> DEFAULT_DISTANCE_FUNCTION = EuclideanDistanceFunction.class;

  /**
   * Parameter for distance function
   */
  public static final OptionID DISTANCE_FUNCTION_ID = new OptionID("rdknn.distancefunction", "Distance function to determine the distance between database objects.");

  /**
   * Parameter k.
   */
  private int k_max;

  /**
   * The distance function.
   */
  private SpatialPrimitiveDistanceFunction<O, D> distanceFunction;

  /**
   * Constructor.
   * 
   * @param pageFileFactory Data storage
   * @param bulkSplitter Bulk loading strategy
   * @param insertionStrategy the strategy to find the insertion child
   * @param k_max
   * @param distanceFunction
   * @param nodeSplitter the strategy for splitting nodes.
   * @param overflowTreatment the strategy to use for overflow treatment
   * @param minimumFill the relative minimum fill
   */
  public RdKNNTreeFactory(PageFileFactory<?> pageFileFactory, AbstractRTreeSettings settings, int k_max, SpatialPrimitiveDistanceFunction<O, D> distanceFunction) {
    super(pageFileFactory, settings);
    this.k_max = k_max;
    this.distanceFunction = distanceFunction;
  }

  @Override
  public RdKNNTree<O, D> instantiate(Relation<O> relation) {
    PageFile<RdKNNNode<D>> pagefile = makePageFile(getNodeClass());
    RdKNNTree<O, D> index = new RdKNNTree<>(relation, pagefile, settings, k_max, distanceFunction, distanceFunction.instantiate(relation));
    return index;
  }

  protected Class<RdKNNNode<D>> getNodeClass() {
    return ClassGenericsUtil.uglyCastIntoSubclass(RdKNNNode.class);
  }

  /**
   * Parameterization class.
   * 
   * @author Erich Schubert
   * 
   * @apiviz.exclude
   */
  public static class Parameterizer<O extends NumberVector<?>, D extends NumberDistance<D, ?>> extends AbstractRStarTreeFactory.Parameterizer<O, AbstractRTreeSettings> {
    /**
     * Parameter k.
     */
    protected int k_max = 0;

    /**
     * The distance function.
     */
    protected SpatialPrimitiveDistanceFunction<O, D> distanceFunction = null;

    @Override
    protected void makeOptions(Parameterization config) {
      super.makeOptions(config);
      IntParameter k_maxP = new IntParameter(K_ID);
      k_maxP.addConstraint(new GreaterConstraint(0));
      if (config.grab(k_maxP)) {
        k_max = k_maxP.intValue();
      }

      ObjectParameter<SpatialPrimitiveDistanceFunction<O, D>> distanceFunctionP = new ObjectParameter<>(DISTANCE_FUNCTION_ID, SpatialPrimitiveDistanceFunction.class, DEFAULT_DISTANCE_FUNCTION);
      if (config.grab(distanceFunctionP)) {
        distanceFunction = distanceFunctionP.instantiateClass(config);
      }
    }

    @Override
    protected RdKNNTreeFactory<O, D> makeInstance() {
      return new RdKNNTreeFactory<>(pageFileFactory, settings, k_max, distanceFunction);
    }

    @Override
    protected AbstractRTreeSettings createSettings() {
      return new AbstractRTreeSettings();
    }
  }
}
