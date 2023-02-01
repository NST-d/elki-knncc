package elki.helper;

import elki.database.ids.DBIDs;

public interface MutualNeighborQuery<O> {

    DBIDs getMutualNeighbors(O query, int k);
}
