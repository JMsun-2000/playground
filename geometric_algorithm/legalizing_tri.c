#include <bits/stdc++.h>
#include <map>
#include <math.h>
using namespace std;

double error = 1e-10;
class Point
{
	public:
		double x, y;
		Point()
		{
			return;
		}
		Point(double x, double y)
		{
			this->x = x;
			this->y = y;
		}
		double distance(Point &b)
		{
			return sqrt(pow((b.x - this->x), 2) + pow((b.y - this->y), 2));
		}
};

class Segment
{
	public:
		//mx + b
		double m, b;
		Point start, end;
		Segment()
		{
			return;
		}
		Segment(Point &start, Point &end)
		{
			this->start = start;
			this->end = end;
			this->m = (end.y - start.y) / (end.x - start.x);
			this->b = start.y - this->m * start.x;
		}
		Segment bisector()
		{
			// this process can be obtained using simple linear equation
			Segment perp;
			Point mid_point((this->start.x + this->end.x) / 2, 
				(this->start.y + this->end.y) / 2);

			perp.m = (double)(-1)/(this->m);

			perp.b = mid_point.y - perp.m * mid_point.x;
			return perp;
		}
		Point intersect(Segment s)
		{
			double x = (s.b - this->b) / (this->m - s.m);
			double y = s.m * x + s.b;
			return Point(x, y);
		}

};
class Circle
{
	public:
		double radius;
		Point center;
		Circle()
		{
			return;
		}
		Circle(Point &center, double radius)
		{
			this->center = center;
			this->radius = radius;
		}
		Circle(Point &a, Point &b, Point &c)
		{
			// to create a circumcircle get the bisector of two segment (3 different points)
			// and intersect them with this you can obtain the center of the circle
			// which is x distance to the points
			Segment s1(a, b), s2(b, c);
			this->center = s1.bisector().intersect(s2.bisector());
			this->radius = this->center.distance(a);
		}
		bool is_inside(Point &a)
		{
			return pow((a.x - this->center.x), 2) + pow((a.y - this->center.y), 2) < pow(this->radius, 2);
		}
};

map<pair<int,int>, vector<int>> triangules;
map<int, Point> points;

void update(int id_begin_point, int id_end_point,
			int id_to_update, int id_to_replace,
			map<pair<int,int>, vector<int>> &not_checked)
{
	// only update the id which not belong to the new triangule
	// formed by the flip this is due to the fact that flip an id
	// change the bottom or the upper point of a triangule
	// try to draw it and you will understand better what i'm trying to explaying

	int min_id=min(id_begin_point, id_end_point);
	int max_id=max(id_begin_point, id_end_point);
	if(triangules[{min_id, max_id}][0] == id_to_replace)
		triangules[{min_id, max_id}][0] = id_to_update;
	else
		triangules[{min_id, max_id}][1] = id_to_update;
	not_checked[{min_id, max_id}] = triangules[{min_id, max_id}];

}
int legalize()
{
	// create a map of every triangule (segment with upper or lower points)
	// which was in the input
	map<pair<int,int>, vector<int>> not_checked(triangules);
	int flips = 0;
	while(!not_checked.empty())
	{
		auto triangule = not_checked.begin();
		// check if this triangule has more than 1 point 
		// if it only has one point it is no necessary to optimize
		// because it is a unique triangule
		if(triangule->second.size() < 2){
			not_checked.erase(triangule);
			continue;
		}
		// the first two are the id points that create the segment which create the triangule
		// the last two are the points upper and lower of that segment 
		//
		int seg_begin_point = triangule->first.first;
		int seg_end_point = triangule->first.second;
		int top_point = triangule->second[0];
		int bottom_point = triangule->second[1];
		// erase the triangule 
		not_checked.erase(triangule);
		// create a circumcircle with the diameter being the segment 
		// that create the triangule and take any of the point
		// could be the top or the bottom it doesn't matter
		Circle circumcircle (points[seg_begin_point], 
							points[seg_end_point], 
							points[top_point]);
		// check if the other point is inside the circle
		if(circumcircle.is_inside(points[bottom_point]))
		{
			// erase the triangule from triangules
			triangules.erase(triangule->first);
			int min_id=min(top_point, bottom_point);
			int max_id=max(top_point, bottom_point);
			// create a new triangule (segment with the condition given below)
			triangules[{min_id,max_id}].push_back(seg_begin_point);
			triangules[{min_id,max_id}].push_back(seg_end_point);
			// update the triangules because we flip a segment
			update(seg_begin_point, top_point, bottom_point, seg_end_point, not_checked);
			update(seg_begin_point, bottom_point, top_point, seg_end_point, not_checked);
			update(seg_end_point, top_point, bottom_point, seg_begin_point, not_checked);
			update(seg_end_point, bottom_point, top_point, seg_begin_point, not_checked);
			flips++;
		}

	}
	return flips;
}
int main()
{
	int n, m;
	cin >> n >> m;
	/*
		general idea:
			every triangle will be represented as a set of segments with the possibility
			to have an upper or a lower point which form the real triangule
			with this is only necessary to check every segment and fliped if neccessary
			and update the corresponding segments which were affected by that flip
				b
			a		c  
				d
			if you have the next segment ({a,b}, {a,d}, {a, c}, {b, c}, {d,c})
			you will have for {a, b} a lower point c, for {c, d} you will have
			an upper point b, and so on (you could have a lower and a upper point
			at the same time but nothing more).
			if you flip the {a, c} segment this is converted in {b, d}
			and for example the {a, b} segment now will have a lower point d instead of c
			and this happend in every segment in the convex hull of this double triangle
	*/
	for(int i=0; i<n; i++)
	{
		double x, y;
		int id;
		cin >> id >> x >> y;
		points[id] = Point(x, y);
	}
	for(int i=0; i<m; i++)
	{
		int id1, id2, id3;
		cin >> id1 >> id2 >> id3;
		triangules[{min(id1, id2), max(id1, id2)}].push_back(id3);
		triangules[{min(id1, id3), max(id1, id3)}].push_back(id2);
		triangules[{min(id2, id3), max(id2, id3)}].push_back(id1);
		
	}
	
	cout << legalize() << endl;

	return 0;
}
